"""Tasks: classical regression and PINN losses sharing the same target function.

Every task exposes:
  d_in, name, order (highest derivative order in the loss), deriv_coord (which input coordinate is differentiated)
  points(N, gen, device) -> dict of tensors (interior/boundary points)
  outputs(model, X) -> dict with f and the derivatives the loss needs (f_x, f_xx, ..., f_t), graph kept
  interior_loss(outs, pts) -> normalized residual/data term computed from `outs`
  boundary_terms(model, pts) -> dict of extra (BC/IC) terms
  loss(model, pts) -> (total, terms, outs)
Losses are normalized by width-independent constants so that they are O(1) at init.
"""
from __future__ import annotations
import math
import torch

PI = math.pi


def grad(y, x, create_graph=True):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=create_graph)[0]


# ----------------------------------------------------------------------- 1D target
# u(x) = sin(2x) + 0.5 cos(3x), x in [-1, 1]
def u1d(x):
    return torch.sin(2 * x) + 0.5 * torch.cos(3 * x)


def u1d_d(x, k):
    s, c = torch.sin, torch.cos
    return {0: lambda: u1d(x),
            1: lambda: 2 * c(2 * x) - 1.5 * s(3 * x),
            2: lambda: -4 * s(2 * x) - 4.5 * c(3 * x),
            3: lambda: -8 * c(2 * x) + 13.5 * s(3 * x),
            4: lambda: 16 * s(2 * x) + 40.5 * c(3 * x)}[k]()


class Task:
    deriv_coord = 0

    def loss(self, model, pts):
        outs = self.outputs(model, pts["X"])
        res = self.interior_loss(outs, pts)
        terms = {"res": res}
        terms.update(self.boundary_terms(model, pts))
        return sum(terms.values()), terms, outs


class Regress1D(Task):
    d_in, order, name = 1, 0, "regress1d"

    def points(self, N, gen, device):
        x = (torch.rand(N, 1, generator=gen) * 2 - 1).to(device)
        return {"X": x, "y": u1d(x)}

    def outputs(self, model, X):
        return {"f": model(X)}

    def interior_loss(self, outs, pts):
        return ((outs["f"] - pts["y"]) ** 2).mean() / pts["y"].pow(2).mean()

    def boundary_terms(self, model, pts):
        return {}


class ODE1D(Task):
    """PINN for d^k u/dx^k = g on [-1,1].  k=1: u(-1) given. k=2: u(+-1). k=4: u(+-1), u'(+-1)."""
    d_in = 1

    def __init__(self, k: int, lam_bc: float = 1.0):
        self.k, self.lam, self.order, self.name = k, lam_bc, k, f"ode{k}"

    def points(self, N, gen, device):
        x = (torch.rand(N, 1, generator=gen) * 2 - 1).to(device)
        xb = torch.tensor([[-1.0], [1.0]], device=device)
        return {"X": x, "g": u1d_d(x, self.k), "xb": xb, "ub": u1d_d(xb, 0), "ub1": u1d_d(xb, 1)}

    def outputs(self, model, X, k=None):
        k = self.k if k is None else k
        X = X.clone().requires_grad_(True)
        f = model(X)
        outs = {"f": f}
        cur = f
        for i in range(1, k + 1):
            cur = grad(cur, X)
            outs["f_" + "x" * i] = cur
        return outs

    def interior_loss(self, outs, pts):
        return ((outs["f_" + "x" * self.k] - pts["g"]) ** 2).mean() / pts["g"].pow(2).mean()

    def boundary_terms(self, model, pts):
        k, xb = self.k, pts["xb"]
        if k == 1:
            bc = ((model(xb[:1]) - pts["ub"][:1]) ** 2).mean()
        elif k == 2:
            bc = ((model(xb) - pts["ub"]) ** 2).mean()
        elif k == 4:
            o = self.outputs(model, xb, k=1)
            bc = ((o["f"] - pts["ub"]) ** 2).mean() + ((o["f_x"] - pts["ub1"]) ** 2).mean() / pts["ub1"].pow(2).mean()
        return {"bc": self.lam * bc}


# ----------------------------------------------------------------------- 2D heat equation
# u_t = u_xx on t in [0,1], x in [-pi, pi]; u = exp(-t) sin(x).
def u_heat(t, x):
    return torch.exp(-t) * torch.sin(x)


class Heat2D(Task):
    """Residual u_t - u_xx, IC u(0,x)=sin x, BC u(t,+-pi)=0. Derivative orders: 1 in t, 2 in x."""
    d_in, order, name, deriv_coord = 2, 2, "heat2d", 1

    def points(self, N, gen, device):
        t = torch.rand(N, 1, generator=gen)
        x = torch.rand(N, 1, generator=gen) * 2 * PI - PI
        xi = torch.rand(N // 4, 1, generator=gen) * 2 * PI - PI
        tb = torch.rand(N // 4, 1, generator=gen)
        X = torch.cat([t, x], 1).to(device)
        Xi = torch.cat([torch.zeros_like(xi), xi], 1).to(device)
        Xb = torch.cat([torch.cat([tb, -PI * torch.ones_like(tb)], 1), torch.cat([tb, PI * torch.ones_like(tb)], 1)], 0).to(device)
        return {"X": X, "Xi": Xi, "ui": torch.sin(xi).to(device), "Xb": Xb,
                "scale": (torch.exp(-2 * t) * torch.sin(x) ** 2).mean().to(device)}

    def outputs(self, model, X):
        X = X.clone().requires_grad_(True)
        f = model(X)
        g1 = grad(f, X)
        f_t, f_x = g1[:, :1], g1[:, 1:]
        f_xx = grad(f_x, X)[:, 1:]
        return {"f": f, "f_t": f_t, "f_x": f_x, "f_xx": f_xx}

    def interior_loss(self, outs, pts):
        return ((outs["f_t"] - outs["f_xx"]) ** 2).mean() / pts["scale"]

    def boundary_terms(self, model, pts):
        ic = ((model(pts["Xi"]) - pts["ui"]) ** 2).mean() / 0.5
        bc = (model(pts["Xb"]) ** 2).mean() / 0.5
        return {"ic": ic, "bc": bc}


class Regress2D(Task):
    d_in, order, name, deriv_coord = 2, 0, "regress2d", 1

    def points(self, N, gen, device):
        t = torch.rand(N, 1, generator=gen)
        x = torch.rand(N, 1, generator=gen) * 2 * PI - PI
        X = torch.cat([t, x], 1).to(device)
        return {"X": X, "y": u_heat(t, x).to(device)}

    def outputs(self, model, X):
        return {"f": model(X)}

    def interior_loss(self, outs, pts):
        return ((outs["f"] - pts["y"]) ** 2).mean() / pts["y"].pow(2).mean()

    def boundary_terms(self, model, pts):
        return {}


TASKS = {
    "regress1d": lambda: Regress1D(),
    "ode1": lambda: ODE1D(1),
    "ode2": lambda: ODE1D(2),
    "ode4": lambda: ODE1D(4),
    "heat2d": lambda: Heat2D(),
    "regress2d": lambda: Regress2D(),
}
