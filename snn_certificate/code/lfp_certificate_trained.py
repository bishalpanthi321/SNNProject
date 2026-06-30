#!/usr/bin/env python3
"""
lfp_certificate_trained.py
==========================
Empirical validation of the Least Fixed-Point (LFP) Spike-Preservation Certificate
for a TRAINED direct-encoded LIF classifier under RESET-BY-SUBTRACTION dynamics.

Difference from the random-weight version: W1 is now the first layer of a small
LIF classifier that is actually TRAINED on the dataset before the certificate is
evaluated. The certificate is then checked on those trained weights.

Two decisive numbers (unchanged):
  1. viol  -> must be 0 (confirms the proof; data/weight-independent)
  2. certified-safe fraction = mean(1 - tau_min) -> usefulness, now on TRAINED weights

Neuron model (reset-by-subtraction):
    V[i,t] = beta*V[i,t-1] + (W1 x)_i - theta*s[i,t-1]
    s[i,t] = 1 if V[i,t] >= theta else 0
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Trainable LIF classifier (surrogate gradient on the spike)
# ----------------------------------------------------------------------
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, theta):
        ctx.save_for_backward(v)
        ctx.theta = theta
        return (v >= theta).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        # triangle surrogate around threshold
        sg = torch.clamp(1.0 - torch.abs(v - ctx.theta), min=0.0)
        return grad_out * sg, None


class LIFClassifier(nn.Module):
    """One hidden LIF layer (the one we certify) + linear readout on spike counts."""
    def __init__(self, in_dim, n_hidden, n_classes=10, beta=0.9, theta=1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, n_hidden, bias=False)
        self.readout = nn.Linear(n_hidden, n_classes)
        self.beta = beta
        self.theta = theta
        self.n_hidden = n_hidden

    def forward(self, x, T):
        drive = self.fc1(x)                      # [B, H], constant across t
        V = torch.zeros(x.shape[0], self.n_hidden, device=x.device)
        s_prev = torch.zeros_like(V)
        spk_sum = torch.zeros_like(V)
        for _ in range(T):
            V = self.beta * V + drive - self.theta * s_prev
            s = SurrogateSpike.apply(V, self.theta)
            spk_sum = spk_sum + s
            s_prev = s
        return self.readout(spk_sum / T)

    @torch.no_grad()
    def run_layer1(self, x, T):
        """Run ONLY the certified hidden layer; record V and s over time (no grad)."""
        drive = self.fc1(x)
        V = torch.zeros(x.shape[0], self.n_hidden, device=x.device)
        s_prev = torch.zeros_like(V)
        V_rec, s_rec = [], []
        for _ in range(T):
            V = self.beta * V + drive - self.theta * s_prev
            s = (V >= self.theta).float()
            V_rec.append(V.clone()); s_rec.append(s.clone())
            s_prev = s
        return torch.stack(V_rec, 2), torch.stack(s_rec, 2)   # [B,H,T] each

    def row_norms(self):
        return torch.linalg.norm(self.fc1.weight.detach(), dim=1)   # [H]


# ----------------------------------------------------------------------
# l2 perturbation with ||delta||_2 = eps
# ----------------------------------------------------------------------
def l2_perturbation(x, eps, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=g, device=x.device)
    flat = noise.view(x.shape[0], -1)
    norm = torch.linalg.norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / norm * eps).view_as(x)


# ----------------------------------------------------------------------
# Least fixed point (upward Kleene iteration), vectorized over neurons
# ----------------------------------------------------------------------
def least_fixed_point(margin, row_norms, beta, theta, eps, T):
    N = margin.shape[0]
    t_idx = torch.arange(1, T + 1, dtype=torch.float64, device=margin.device)
    B_input = (eps * row_norms.double().unsqueeze(1)
               * ((1.0 - beta ** t_idx) / (1.0 - beta)).unsqueeze(0))   # [N,T]
    m = margin.double()
    tau = torch.zeros(N, T, dtype=torch.float64, device=margin.device)
    for _ in range(T + 1):
        reset_term = torch.zeros(N, T, dtype=torch.float64, device=margin.device)
        for t0 in range(1, T):
            j = torch.arange(0, t0, device=margin.device)
            coeff = theta * (beta ** (t0 - 1 - j).double())
            reset_term[:, t0] = (tau[:, :t0] * coeff.unsqueeze(0)).sum(dim=1)
        total = B_input + reset_term
        tau_next = (m <= total).double()
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
        Xtr = torch.rand(512, in_dim); Ytr = torch.randint(0, 10, (512,))
        Xte = torch.rand(128, in_dim); Yte = torch.randint(0, 10, (128,))
        tr = torch.utils.data.TensorDataset(Xtr, Ytr)
        te = torch.utils.data.TensorDataset(Xte, Yte)
        return (torch.utils.data.DataLoader(tr, batch, shuffle=True),
                torch.utils.data.DataLoader(te, batch, shuffle=False), in_dim)
    from torchvision import datasets, transforms
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    DS = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST,
          "cifar10": datasets.CIFAR10}[dataset]
    tr = DS("./data", train=True, download=True, transform=tfm)
    te = DS("./data", train=False, download=True, transform=tfm)
    flat = lambda b: (b[0].view(b[0].size(0), -1), b[1])
    return (torch.utils.data.DataLoader(tr, batch, shuffle=True, collate_fn=None),
            torch.utils.data.DataLoader(te, batch, shuffle=False), in_dim)


def train_model(model, loader, device, T, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb = xb.view(xb.size(0), -1).to(device); yb = yb.to(device)
            loss = F.cross_entropy(model(xb, T), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"      epoch {ep+1}/{epochs} done", flush=True)
    model.eval()


# ----------------------------------------------------------------------
# One configuration
# ----------------------------------------------------------------------
def run_config(dataset, beta, T, args, device):
    tr, te, in_dim = get_loaders(dataset, args.batch, fake=args.fake)
    model = LIFClassifier(in_dim, args.n_hidden, beta=beta, theta=args.theta).to(device)
    print(f"    training {dataset} beta={beta} T={T} ...", flush=True)
    train_model(model, tr, device, T, args.epochs)

    # gather a test batch of samples to certify
    xb, _ = next(iter(te))
    xb = xb.view(xb.size(0), -1).to(device)[:args.n_samples]

    viol_total = 0; safe_fracs = []; flip_fracs = []
    for idx in range(xb.shape[0]):
        x = xb[idx:idx+1]
        delta = l2_perturbation(x, args.eps, seed=args.seed + idx)
        Vc, sc = model.run_layer1(x, T)
        Vp, sp = model.run_layer1(x + delta, T)
        Vc, sc, Vp, sp = Vc[0], sc[0], Vp[0], sp[0]          # [H,T]

        e_true = (sp - sc).abs()
        margin = (Vc - model.theta).abs()
        tau = least_fixed_point(margin, model.row_norms(), beta, args.theta, args.eps, T)

        bad = ((tau == 0) & (e_true.double() == 1))
        n_bad = int(bad.sum().item())
        if n_bad > 0:
            ii, tt = torch.where(bad)
            raise AssertionError(
                f"SOUNDNESS VIOLATION ds={dataset} beta={beta} T={T} sample={idx}: "
                f"{n_bad} cases, first neuron={ii[0].item()} t={tt[0].item()}")
        viol_total += n_bad
        safe_fracs.append(float((1.0 - tau).mean().item()))
        flip_fracs.append(float(e_true.mean().item()))

    return {"dataset": dataset, "beta": beta, "T": T, "viol": viol_total,
            "safe_frac": float(np.mean(safe_fracs)),
            "flip_frac": float(np.mean(flip_fracs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashion", "cifar10"])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.8, 0.9, 0.99])
    ap.add_argument("--timesteps", nargs="+", type=int, default=[10, 20, 50])
    ap.add_argument("--n-hidden", type=int, default=128)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"LFP certificate validation (TRAINED) | device={device} hidden={args.n_hidden} "
          f"eps={args.eps} theta={args.theta} epochs={args.epochs} samples={args.n_samples}",
          flush=True)
    print(f"reset = subtraction | datasets={args.datasets} betas={args.betas} "
          f"T={args.timesteps}\n", flush=True)

    results = []
    for ds in args.datasets:
        for beta in args.betas:
            for T in args.timesteps:
                r = run_config(ds, beta, T, args, device)
                results.append(r)
                print(f"  {ds:8s} beta={beta:<4} T={T:<3} | viol={r['viol']:<4} | "
                      f"certified-safe={100*r['safe_frac']:5.1f}% | "
                      f"flip-rate={100*r['flip_frac']:4.1f}%\n", flush=True)

    print("=== SOUNDNESS ===", flush=True)
    total_viol = sum(r["viol"] for r in results)
    print(f"Total violations across {len(results)} configs: {total_viol}  "
          f"({'PASS - proof confirmed on trained weights' if total_viol == 0 else 'FAIL'})",
          flush=True)

    print("\n=== CERTIFIED-SAFE FRACTION (rows=beta, cols=T) ===", flush=True)
    for ds in args.datasets:
        print(f"\n[{ds}]", flush=True)
        print("beta\\T  " + "".join(f"{T:>8}" for T in args.timesteps), flush=True)
        for beta in args.betas:
            row = f"{beta:<6}  "
            for T in args.timesteps:
                r = next(x for x in results if x["dataset"] == ds
                         and x["beta"] == beta and x["T"] == T)
                row += f"{100*r['safe_frac']:7.1f}%"
            print(row, flush=True)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
