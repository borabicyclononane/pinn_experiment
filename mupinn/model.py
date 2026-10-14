"""MLP with explicit width parametrization (SP / NTK / muP) and per-layer LR multipliers.

Conventions (constant input dim d, width n, base width n0 for LR multipliers):

  h1 = W1 x + b1,  x_l = phi(h_l),  h_{l+1} = W_{l+1} x_l + b_{l+1},  f = W_out x_L + b_out

Parametrization table (init std, SGD lr multiplier, Adam lr multiplier), written so that
all parametrizations coincide at n = n0 except for the output-layer init in muP:

  SP   : W1 ~ 1/sqrt(d), Wh ~ 1/sqrt(n), Wout ~ 1/sqrt(n);     lr mult 1 everywhere (both optimizers)
  NTK  : same init as SP;  SGD mult (1, n0/n, n0/n) ; Adam mult (1, n0/n, n0/n)   [not used in main text]
  muP  : W1 ~ 1/sqrt(d), Wh ~ 1/sqrt(n), Wout ~ 1/n;
         SGD mult (n/n0, 1, n0/n),  Adam mult (1, n0/n, n0/n)
  Hidden biases follow the input-layer rule (fan_in = 1, fan_out = n); the output bias has fan_out = 1 and
  therefore a width-free learning rate (getting this wrong makes delta f grow linearly in n).

Everything is implemented with plain parameters and per-parameter-group learning rates
(no forward multipliers), i.e. exactly Table 3 of Yang et al. 2022 / Definition 1 of the
18.S995 notes.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

ACTS = {
    "tanh": torch.tanh,
    "sin": torch.sin,
    "gelu": torch.nn.functional.gelu,
    "relu": torch.relu,
}


def _tanh_d(h):
    t = torch.tanh(h); return 1 - t * t


def _tanh_dd(h):
    t = torch.tanh(h); return -2 * t * (1 - t * t)


ACT_DERIVS = {
    "tanh": (torch.tanh, _tanh_d, _tanh_dd),
    "sin": (torch.sin, torch.cos, lambda h: -torch.sin(h)),
}


class MLP(nn.Module):
    def __init__(self, d_in: int, width: int, depth: int, param: str = "mup",
                 act: str = "tanh", base_width: int = 64, out_init: str = "default",
                 bias_std_in: float = 1.0, seed: int = 0, dtype=torch.float32):
        """depth = number of hidden layers (L). Layers: W1 (d->n), W2..WL (n->n), Wout (n->1)."""
        super().__init__()
        assert param in ("sp", "ntk", "mup")
        self.d_in, self.n, self.L, self.param, self.n0 = d_in, width, depth, param, base_width
        self.act_name = act
        self.act = ACTS[act]
        g = torch.Generator().manual_seed(seed)
        n, d = width, d_in

        def rnd(*shape, std):
            return nn.Parameter((torch.randn(*shape, generator=g, dtype=dtype) * std))

        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        # input layer
        self.W.append(rnd(n, d, std=1.0 / math.sqrt(d)))
        self.b.append(rnd(n, std=bias_std_in))
        # hidden layers
        for _ in range(depth - 1):
            self.W.append(rnd(n, n, std=1.0 / math.sqrt(n)))
            self.b.append(nn.Parameter(torch.zeros(n, dtype=dtype)))
        # output layer
        if out_init == "zero":
            out_std = 0.0
        elif param == "mup":
            out_std = 1.0 / n
        else:
            out_std = 1.0 / math.sqrt(n)
        self.W.append(rnd(1, n, std=out_std) if out_std > 0 else nn.Parameter(torch.zeros(1, n, dtype=dtype)))
        self.b.append(nn.Parameter(torch.zeros(1, dtype=dtype)))

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        hs = []
        h = x @ self.W[0].T + self.b[0]
        hs.append(h)
        a = self.act(h)
        for l in range(1, self.L):
            h = a @ self.W[l].T + self.b[l]
            hs.append(h)
            a = self.act(h)
        f = a @ self.W[self.L].T + self.b[self.L]
        if return_hidden:
            return f, hs
        return f

    # ------------------------------------------------------------------ forward-mode tangents
    def forward_tangent(self, x: torch.Tensor, coord: int = 0):
        """Forward-mode propagation of per-neuron first and second derivatives w.r.t. input coordinate `coord`.
        Returns (f, hs, vs, as_): hs[l] = h_l (N,n), vs[l] = d h_l / d x_c (N,n), as_[l] = d^2 h_l / d x_c^2 (N,n)."""
        act, d1, d2 = ACT_DERIVS[self.act_name]
        h = x @ self.W[0].T + self.b[0]
        v = self.W[0][:, coord].unsqueeze(0).expand_as(h)
        a = torch.zeros_like(h)
        hs, vs, as_ = [h], [v], [a]
        for l in range(1, self.L + 1):
            xl, dx, ddx = act(h), d1(h) * v, d2(h) * v * v + d1(h) * a
            h = xl @ self.W[l].T + self.b[l]
            v = dx @ self.W[l].T
            a = ddx @ self.W[l].T
            hs.append(h); vs.append(v); as_.append(a)
        return hs[-1], hs[:-1], vs[:-1], as_[:-1]

    # ------------------------------------------------------------------ lr groups
    def lr_multipliers(self, optimizer: str):
        """Per-layer LR multipliers for layers [W1, W2..WL, Wout] (biases follow the input rule)."""
        n, n0, p = self.n, self.n0, self.param
        r = n / n0
        if p == "sp":
            win, wh, wout = 1.0, 1.0, 1.0
        elif p == "ntk":
            win, wh, wout = 1.0, 1.0 / r, 1.0 / r
        elif p == "mup":
            if optimizer == "sgd":
                win, wh, wout = r, 1.0, 1.0 / r
            else:  # adam / adamw
                win, wh, wout = 1.0, 1.0 / r, 1.0 / r
        mults = [win] + [wh] * (self.L - 1) + [wout]
        return mults, win  # bias multiplier = input rule

    def param_groups(self, lr: float, optimizer: str):
        mults, bmult = self.lr_multipliers(optimizer)
        groups = []
        for l in range(self.L + 1):
            groups.append({"params": [self.W[l]], "lr": lr * mults[l], "name": f"W{l+1}" if l < self.L else "Wout", "mult": mults[l]})
            # biases: fan_in = 1, fan_out = width for hidden biases (input rule), but the OUTPUT bias has
            # fan_out = 1 and enters f directly, so its learning rate carries no width factor.
            bm = bmult if l < self.L else 1.0
            groups.append({"params": [self.b[l]], "lr": lr * bm, "name": f"b{l+1}" if l < self.L else "bout", "mult": bm})
        return groups

    def make_optimizer(self, lr: float, optimizer: str = "sgd", **kw):
        groups = self.param_groups(lr, optimizer)
        if optimizer == "sgd":
            return torch.optim.SGD(groups, lr=lr, **kw)
        if optimizer == "adam":
            return torch.optim.Adam(groups, lr=lr, **kw)
        raise ValueError(optimizer)

    def layer_names(self):
        return [f"W{l+1}" for l in range(self.L)] + ["Wout"]
