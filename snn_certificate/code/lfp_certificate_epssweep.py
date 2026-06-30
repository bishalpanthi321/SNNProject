#!/usr/bin/env python3
"""
lfp_certificate_epssweep.py
===========================
Epsilon-sweep stress test of the Least Fixed-Point (LFP) Spike-Preservation
Certificate on TRAINED LIF classifiers under RESET-BY-SUBTRACTION.

Key question this answers
-------------------------
How does the certified-safe fraction DEGRADE as the perturbation radius eps grows?
A robustness certificate that only works at tiny eps is weak; we want the full
curve: certified-safe(eps) and flip-rate(eps).

Design: train ONCE per (dataset, beta, T); evaluate the certificate at EVERY eps
in the sweep on that same trained model (weights don't depend on eps).

Two checks per (config, eps):
  1. viol == 0           (soundness; must hold at every eps)
  2. certified-safe frac (usefulness; expected to fall as eps rises)

Optional: random (--attack gaussian, default) or FGSM (--attack fgsm) perturbation.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Surrogate-gradient spike + trainable LIF classifier
# ----------------------------------------------------------------------
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, theta):
        ctx.save_for_backward(v); ctx.theta = theta
        return (v >= theta).float()
    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        sg = torch.clamp(1.0 - torch.abs(v - ctx.theta), min=0.0)
        return g * sg, None


class LIFClassifier(nn.Module):
    def __init__(self, in_dim, n_hidden, n_classes=10, beta=0.9, theta=1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, n_hidden, bias=False)
        self.readout = nn.Linear(n_hidden, n_classes)
        self.beta = beta; self.theta = theta; self.n_hidden = n_hidden

    def forward(self, x, T):
        drive = self.fc1(x)
        V = torch.zeros(x.shape[0], self.n_hidden, device=x.device)
        s_prev = torch.zeros_like(V); spk = torch.zeros_like(V)
        for _ in range(T):
            V = self.beta * V + drive - self.theta * s_prev
            s = SurrogateSpike.apply(V, self.theta)
            spk = spk + s; s_prev = s
        return self.readout(spk / T)

    @torch.no_grad()
    def run_layer1(self, x, T):
        drive = self.fc1(x)
        V = torch.zeros(x.shape[0], self.n_hidden, device=x.device)
        s_prev = torch.zeros_like(V); V_rec, s_rec = [], []
        for _ in range(T):
            V = self.beta * V + drive - self.theta * s_prev
            s = (V >= self.theta).float()
            V_rec.append(V.clone()); s_rec.append(s.clone()); s_prev = s
        return torch.stack(V_rec, 2), torch.stack(s_rec, 2)

    def row_norms(self):
        return torch.linalg.norm(self.fc1.weight.detach(), dim=1)


# ----------------------------------------------------------------------
# Perturbations
# ----------------------------------------------------------------------
def l2_gaussian(x, eps, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    n = torch.randn(x.shape, generator=g, device=x.device)
    flat = n.view(x.shape[0], -1)
    nrm = torch.linalg.norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / nrm * eps).view_as(x)

def l2_fgsm(model, x, y, eps, T):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x, T), y)
    grad = torch.autograd.grad(loss, x)[0]
    flat = grad.view(x.shape[0], -1)
    nrm = torch.linalg.norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / nrm * eps).view_as(x).detach()   # steepest L2 step, ||.||=eps


# ----------------------------------------------------------------------
# Least fixed point
# ----------------------------------------------------------------------
def least_fixed_point(margin, row_norms, beta, theta, eps, T):
    N = margin.shape[0]
    t_idx = torch.arange(1, T + 1, dtype=torch.float64, device=margin.device)
    B_input = (eps * row_norms.double().unsqueeze(1)
               * ((1.0 - beta ** t_idx) / (1.0 - beta)).unsqueeze(0))
    m = margin.double()
    tau = torch.zeros(N, T, dtype=torch.float64, device=margin.device)
    for _ in range(T + 1):
        reset_term = torch.zeros(N, T, dtype=torch.float64, device=margin.device)
        for t0 in range(1, T):
            j = torch.arange(0, t0, device=margin.device)
            coeff = theta * (beta ** (t0 - 1 - j).double())
            reset_term[:, t0] = (tau[:, :t0] * coeff.unsqueeze(0)).sum(dim=1)
        tau_next = (m <= B_input + reset_term).double()
        if torch.equal(tau_next, tau):
            tau = tau_next; break
        tau = tau_next
    return tau


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def get_loaders(dataset, batch, fake=False):
    in_dim = 3072 if dataset == "cifar10" else 784
    if fake:
        tr = torch.utils.data.TensorDataset(torch.rand(512, in_dim), torch.randint(0, 10, (512,)))
        te = torch.utils.data.TensorDataset(torch.rand(128, in_dim), torch.randint(0, 10, (128,)))
        return (torch.utils.data.DataLoader(tr, batch, shuffle=True),
                torch.utils.data.DataLoader(te, batch, shuffle=False), in_dim)
    from torchvision import datasets, transforms
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    DS = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}[dataset]
    tr = DS("./data", train=True, download=True, transform=tfm)
    te = DS("./data", train=False, download=True, transform=tfm)
    return (torch.utils.data.DataLoader(tr, batch, shuffle=True),
            torch.utils.data.DataLoader(te, batch, shuffle=False), in_dim)

def train_model(model, loader, device, T, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb = xb.view(xb.size(0), -1).to(device); yb = yb.to(device)
            loss = F.cross_entropy(model(xb, T), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"      epoch {ep+1}/{epochs} done", flush=True)
    model.eval()


# ----------------------------------------------------------------------
# Evaluate certificate at one eps (model already trained)
# ----------------------------------------------------------------------
def eval_eps(model, xb, yb, beta, T, eps, args, device):
    viol_total = 0; safe_fracs = []; flip_fracs = []
    for idx in range(xb.shape[0]):
        x = xb[idx:idx+1]
        if args.attack == "fgsm":
            y = yb[idx:idx+1]
            delta = l2_fgsm(model, x, y, eps, T)
        else:
            delta = l2_gaussian(x, eps, seed=args.seed + idx)
        Vc, sc = model.run_layer1(x, T)
        Vp, sp = model.run_layer1(x + delta, T)
        Vc, sc, Vp, sp = Vc[0], sc[0], Vp[0], sp[0]
        e_true = (sp - sc).abs()
        margin = (Vc - model.theta).abs()
        tau = least_fixed_point(margin, model.row_norms(), beta, args.theta, eps, T)
        bad = ((tau == 0) & (e_true.double() == 1))
        n_bad = int(bad.sum().item())
        if n_bad > 0:
            ii, tt = torch.where(bad)
            raise AssertionError(
                f"SOUNDNESS VIOLATION ds={args._ds} beta={beta} T={T} eps={eps} "
                f"sample={idx}: first neuron={ii[0].item()} t={tt[0].item()}")
        viol_total += n_bad
        safe_fracs.append(float((1.0 - tau).mean().item()))
        flip_fracs.append(float(e_true.mean().item()))
    return viol_total, float(np.mean(safe_fracs)), float(np.mean(flip_fracs))


# ----------------------------------------------------------------------
# One (dataset, beta, T): train once, sweep eps
# ----------------------------------------------------------------------
def run_config(dataset, beta, T, args, device):
    args._ds = dataset
    tr, te, in_dim = get_loaders(dataset, args.batch, fake=args.fake)
    model = LIFClassifier(in_dim, args.n_hidden, beta=beta, theta=args.theta).to(device)
    print(f"    training {dataset} beta={beta} T={T} ...", flush=True)
    train_model(model, tr, device, T, args.epochs)
    xb, yb = next(iter(te))
    xb = xb.view(xb.size(0), -1).to(device)[:args.n_samples]
    yb = yb.to(device)[:args.n_samples]
    rows = []
    for eps in args.eps_sweep:
        v, sf, ff = eval_eps(model, xb, yb, beta, T, eps, args, device)
        rows.append({"dataset": dataset, "beta": beta, "T": T, "eps": eps,
                     "viol": v, "safe_frac": sf, "flip_frac": ff})
        print(f"  {dataset:8s} beta={beta:<4} T={T:<3} eps={eps:<4} | "
              f"viol={v:<3} | certified-safe={100*sf:5.1f}% | flip-rate={100*ff:4.1f}%",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashion", "cifar10"])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.9])
    ap.add_argument("--timesteps", nargs="+", type=int, default=[20])
    ap.add_argument("--eps-sweep", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--attack", choices=["gaussian", "fgsm"], default="gaussian")
    ap.add_argument("--n-hidden", type=int, default=128)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"LFP eps-sweep (TRAINED) | device={device} hidden={args.n_hidden} "
          f"attack={args.attack} epochs={args.epochs} samples={args.n_samples}", flush=True)
    print(f"reset=subtraction | datasets={args.datasets} betas={args.betas} "
          f"T={args.timesteps} eps_sweep={args.eps_sweep}\n", flush=True)

    all_rows = []
    for ds in args.datasets:
        for beta in args.betas:
            for T in args.timesteps:
                all_rows.extend(run_config(ds, beta, T, args, device))
                print("", flush=True)

    # soundness
    print("=== SOUNDNESS ===", flush=True)
    tv = sum(r["viol"] for r in all_rows)
    print(f"Total violations across {len(all_rows)} (config,eps) points: {tv}  "
          f"({'PASS' if tv == 0 else 'FAIL'})\n", flush=True)

    # degradation curve table: certified-safe % vs eps, per (dataset,beta,T)
    print("=== CERTIFIED-SAFE FRACTION vs EPS  (degradation curve) ===", flush=True)
    header = "config".ljust(26) + "".join(f"{e:>8}" for e in args.eps_sweep)
    print(header, flush=True)
    seen = []
    for ds in args.datasets:
        for beta in args.betas:
            for T in args.timesteps:
                key = (ds, beta, T)
                if key in seen: continue
                seen.append(key)
                label = f"{ds},b={beta},T={T}".ljust(26)
                line = label
                for e in args.eps_sweep:
                    r = next(x for x in all_rows if x["dataset"] == ds and x["beta"] == beta
                             and x["T"] == T and x["eps"] == e)
                    line += f"{100*r['safe_frac']:7.1f}%"
                print(line, flush=True)

    # also dump flip-rate vs eps (context for how hard the attack got)
    print("\n=== FLIP-RATE vs EPS ===", flush=True)
    print(header, flush=True)
    seen = []
    for ds in args.datasets:
        for beta in args.betas:
            for T in args.timesteps:
                key = (ds, beta, T)
                if key in seen: continue
                seen.append(key)
                label = f"{ds},b={beta},T={T}".ljust(26)
                line = label
                for e in args.eps_sweep:
                    r = next(x for x in all_rows if x["dataset"] == ds and x["beta"] == beta
                             and x["T"] == T and x["eps"] == e)
                    line += f"{100*r['flip_frac']:7.1f}%"
                print(line, flush=True)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
