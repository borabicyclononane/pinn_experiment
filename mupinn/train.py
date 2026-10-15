"""Training loop with per-step instrumentation (coordinate-check statistics).

At each measured step t (state after t updates) and for every node
(x, h1..hL, f, dx_h_l, dxx_h_l, f_x, f_xx, ..., and weights W1..Wout):
  value : RMS over batch & coordinates
  delta : RMS of (node - node at init)
  grad  : RMS of d(loss)/d(node)
  gradf : RMS of d(mean f)/d(node)
  update: RMS of the parameter update applied in the step that produced this state (weights only)
  grad_<term>: RMS of d(term)/d(weights) for every loss term
"""
from __future__ import annotations
import math, time
import torch
from .model import MLP
from .tasks import TASKS, grad


def rms(t):
    return float(t.detach().float().pow(2).mean().sqrt())


def log_steps(T):
    s = {0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384}
    return sorted(x for x in s if x <= T) + ([T] if T not in s else [])


def forward_with_nodes(model, task, pts, max_order=None):
    """One forward pass that produces the loss AND all measured intermediate nodes on the same graph."""
    X = pts["X"].clone().requires_grad_(True)
    f, hs = model(X, return_hidden=True)
    nodes = {"x": X}
    for l, h in enumerate(hs):
        nodes[f"h{l+1}"] = h
    nodes["f"] = f
    c = task.deriv_coord
    order = max(task.order, 1) if max_order is None else max_order
    # per-neuron tangent nodes for hidden layers via forward mode (value/delta only; detached from the loss graph)
    with torch.no_grad():
        _, _, vs, as_ = model.forward_tangent(X.detach(), coord=c)
    for l in range(len(hs)):
        nodes[f"dx_h{l+1}"] = vs[l]
        if order >= 2 and l >= 1:
            nodes[f"dxx_h{l+1}"] = as_[l]
    outs = {"f": f}
    cur = f
    if task.d_in == 2:
        g1 = grad(f, X)
        outs["f_t"] = g1[:, :1]
        cur = g1[:, c:c + 1]
        outs["f_x"] = cur
        nodes["f_t"] = outs["f_t"]
        nodes["f_x"] = cur
        for k in range(2, order + 1):
            cur = grad(cur, X)[:, c:c + 1]
            outs["f_" + "x" * k] = cur
            nodes["f_" + "x" * k] = cur
    else:
        for k in range(1, order + 1):
            cur = grad(cur, X)
            outs["f_" + "x" * k] = cur
            nodes["f_" + "x" * k] = cur
    res = task.interior_loss(outs, pts)
    terms = {"res": res}
    terms.update(task.boundary_terms(model, pts))
    total = sum(terms.values())
    return total, terms, nodes


def measure(model, task, pts, init_nodes, init_params, applied_update, want_grad=True):
    total, terms, nodes = forward_with_nodes(model, task, pts)
    stats = {}
    for k, v in nodes.items():
        stats[f"{k}/value"] = rms(v)
        if init_nodes is not None and k in init_nodes:
            stats[f"{k}/delta"] = rms(v - init_nodes[k])
    names = model.layer_names()
    for l, p in enumerate(model.W):
        stats[f"{names[l]}/value"] = rms(p)
        if init_params is not None:
            stats[f"{names[l]}/delta"] = rms(p - init_params[l])
        if applied_update is not None:
            stats[f"{names[l]}/update"] = rms(applied_update[l])
    if want_grad:
        gnodes = {k: v for k, v in nodes.items() if v.requires_grad}
        node_list = list(gnodes.values())
        g_nodes = torch.autograd.grad(total, node_list + list(model.W), retain_graph=True, allow_unused=True)
        for (k, _), g in zip(gnodes.items(), g_nodes[:len(gnodes)]):
            stats[f"{k}/grad"] = rms(g) if g is not None else float("nan")
        for l, g in enumerate(g_nodes[len(gnodes):]):
            stats[f"{names[l]}/grad"] = rms(g) if g is not None else float("nan")
        fmean = nodes["f"].mean()
        g_f = torch.autograd.grad(fmean, node_list + list(model.W), retain_graph=True, allow_unused=True)
        for (k, _), g in zip(gnodes.items(), g_f[:len(gnodes)]):
            stats[f"{k}/gradf"] = rms(g) if g is not None else float("nan")
        for l, g in enumerate(g_f[len(gnodes):]):
            stats[f"{names[l]}/gradf"] = rms(g) if g is not None else float("nan")
        if len(terms) > 1:
            for tname, tval in terms.items():
                gt = torch.autograd.grad(tval, list(model.W), retain_graph=True, allow_unused=True)
                for l, g in enumerate(gt):
                    stats[f"{names[l]}/grad_{tname}"] = rms(g) if g is not None else float("nan")
    return stats, float(total), {k: float(v) for k, v in terms.items()}, {k: v.detach().clone() for k, v in nodes.items()}


def rel_l2_error(model, task, device):
    with torch.no_grad():
        if task.d_in == 1:
            from .tasks import u1d
            xs = torch.linspace(-1, 1, 2001, device=device)[:, None]
            u = u1d(xs)
            return float(((model(xs) - u) ** 2).mean().sqrt() / u.pow(2).mean().sqrt())
        from .tasks import u_heat
        g = torch.Generator().manual_seed(7)
        tt = torch.rand(4000, 1, generator=g); xx = torch.rand(4000, 1, generator=g) * 2 * math.pi - math.pi
        X = torch.cat([tt, xx], 1).to(device)
        u = u_heat(tt, xx).to(device)
        return float(((model(X) - u) ** 2).mean().sqrt() / u.pow(2).mean().sqrt())


def train(task_name: str, width: int, depth: int, param: str, lr: float, steps: int,
          optimizer: str = "sgd", act: str = "tanh", seed: int = 0, N: int = 512,
          device: str = "cpu", measure_steps=None, want_grad=True,
          base_width: int = 64, out_init: str = "default", data_seed: int = 1234,
          eval_pts: int = 1000, time_limit: float = None, adam_betas=(0.9, 0.999), verbose=False,
          div_thresh: float = 1e4, resample: bool = False):
    """Full-batch training on fixed collocation points (resample=True draws fresh points every step; the
    measured statistics always use the fixed initial batch)."""
    torch.manual_seed(seed)
    task = TASKS[task_name]()
    gen = torch.Generator().manual_seed(data_seed)
    pts = task.points(N, gen, device)
    gen_train = torch.Generator().manual_seed(data_seed + 7 + seed)
    pts_eval = task.points(eval_pts, torch.Generator().manual_seed(data_seed + 1), device)
    model = MLP(task.d_in, width, depth, param=param, act=act, base_width=base_width, out_init=out_init, seed=seed).to(device)
    opt = model.make_optimizer(lr, optimizer, betas=adam_betas) if optimizer == "adam" else model.make_optimizer(lr, optimizer)
    measure_steps = set(log_steps(steps) if measure_steps is None else measure_steps) | {steps}
    init_params = [p.detach().clone() for p in model.W]
    init_nodes = None
    hist = {"step": [], "loss": [], "terms": [], "eval_loss": []}
    stats_at = {}
    t0 = time.time()
    applied_update = None
    diverged = False
    best_eval = float("inf")
    for t in range(steps + 1):
        if t in measure_steps:
            st, L, terms, nodes = measure(model, task, pts, init_nodes, init_params, applied_update, want_grad=want_grad)
            if init_nodes is None:
                init_nodes = nodes
            Le_t, _, _ = task.loss(model, pts_eval)
            Le = float(Le_t)
            stats_at[t] = st
            hist["step"].append(t); hist["loss"].append(L); hist["terms"].append(terms); hist["eval_loss"].append(Le)
            best_eval = min(best_eval, Le) if math.isfinite(Le) else best_eval
            if verbose:
                print(f"t={t:5d} loss={L:.4e} eval={Le:.4e} " + " ".join(f"{k}={v:.3e}" for k, v in terms.items()), flush=True)
            if not math.isfinite(L) or L > div_thresh:
                diverged = True
                break
        if t == steps:
            break
        opt.zero_grad(set_to_none=True)
        total, terms, _ = task.loss(model, task.points(N, gen_train, device) if resample else pts)
        if not math.isfinite(float(total)):
            diverged = True
            break
        total.backward()
        need = (t + 1) in measure_steps
        before = [p.detach().clone() for p in model.W] if need else None
        opt.step()
        applied_update = [p.detach() - b for p, b in zip(model.W, before)] if need else None
        if time_limit and time.time() - t0 > time_limit:
            break
    err = rel_l2_error(model, task, device) if not diverged else float("nan")
    return {"task": task_name, "width": width, "depth": depth, "param": param, "lr": lr, "steps": steps,
            "optimizer": optimizer, "act": act, "seed": seed, "N": N, "hist": hist, "stats": stats_at,
            "final_loss": hist["loss"][-1], "final_eval": hist["eval_loss"][-1], "best_eval": best_eval,
            "rel_l2": err, "diverged": diverged, "time": time.time() - t0}
