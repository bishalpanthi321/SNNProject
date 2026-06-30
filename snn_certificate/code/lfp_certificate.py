#!/usr/bin/env python3
"""
lfp_certificate.py
==================
Empirical validation of the Least Fixed-Point (LFP) Spike-Preservation Certificate
for a direct-encoded LIF layer under RESET-BY-SUBTRACTION dynamics.

What this script verifies
-------------------------
The companion proof claims: under the least fixed point tau_min of the causal
operator F, every neuron flagged SAFE (tau=0) provably does not flip
(viol = 0). This script tests that claim empirically AND measures whether the
certificate is *useful* (the certified-safe fraction).

Two numbers decide everything:
  1. viol  -> must be 0 (confirms the proof; if >0, the proof or the code is wrong)
  2. certified-safe fraction = mean(1 - tau_min)  -> the precision / usefulness

Neuron model (reset-by-subtraction, applied inside the update loop):
    V[i,t] = beta*V[i,t-1] + (W1 x)_i - theta*s[i,t-1]
    s[i,t] = 1 if V[i,t] >= theta else 0

Certificate bound (causal, per neuron, per timestep):
    B_input(i,t)   = eps * ||row_i(W1)||_2 * (1 - beta^t)/(1 - beta)
    reset_term(i,t)= theta * sum_{k=1}^{t} beta^(t-k) * tau[i, k-1]
    B_reset(i,t)   = B_input(i,t) + reset_term(i,t)
    F(tau)[i,t]    = 1[ m(i,t) <= B_reset(i,t) ]      (NON-strict, matches proof)

Least fixed point via upward Kleene iteration from tau=0.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# LIF layer with reset-by-subtraction; records V and s over time
# ----------------------------------------------------------------------
class LIFLayer(nn.Module):
    def __init__(self, in_dim, n_hidden, beta=0.9, theta=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(n_hidden, in_dim, generator=g) / np.sqrt(in_dim),
                               requires_grad=False)
        self.beta = beta
        self.theta = theta
        self.n_hidden = n_hidden

    @torch.no_grad()
    def run(self, x, T):
        """
        x: [B, in_dim]  (direct encoding: same x injected every timestep)
        returns V: [B, n_hidden, T], s: [B, n_hidden, T]
        """
        B = x.shape[0]
        drive = x @ self.W1.t()                       # [B, n_hidden], constant in t
        V = torch.zeros(B, self.n_hidden, device=x.device)
        s_prev = torch.zeros(B, self.n_hidden, device=x.device)
        V_rec, s_rec = [], []
        for t in range(T):
            # reset-by-subtraction uses PREVIOUS spike
            V = self.beta * V + drive - self.theta * s_prev
            s = (V >= self.theta).float()
            V_rec.append(V.clone())
            s_rec.append(s.clone())
            s_prev = s
        V = torch.stack(V_rec, dim=2)                 # [B, n_hidden, T]
        s = torch.stack(s_rec, dim=2)                 # [B, n_hidden, T]
        return V, s

    def row_norms(self):
        return torch.linalg.norm(self.W1, dim=1)      # [n_hidden]


# ----------------------------------------------------------------------
# l2-bounded perturbation, exactly ||delta||_2 = eps  (rescaled noise)
# ----------------------------------------------------------------------
def l2_perturbation(x, eps, seed=0):
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(x.shape, generator=g)
    flat = noise.view(x.shape[0], -1)
    norm = torch.linalg.norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
    flat = flat / norm * eps
    return flat.view_as(x)


# ----------------------------------------------------------------------
# Least Fixed-Point via upward Kleene iteration (vectorized over neurons)
# ----------------------------------------------------------------------
def least_fixed_point(margin, row_norms, beta, theta, eps, T):
    """
    margin:    [N, T]   m(i,t) = |V_clean - theta|
    row_norms: [N]
    returns tau_min: [N, T] in {0,1}
    """
    N = margin.shape[0]
    # geometric coefficients
    t_idx = torch.arange(1, T + 1, dtype=torch.float64)
    B_input = (eps * row_norms.double().unsqueeze(1)
               * ((1.0 - beta ** t_idx) / (1.0 - beta)).unsqueeze(0))   # [N, T]

    # precompute decay matrix D[t,k] = beta^(t-k) for k<=t else 0   (k,t are 0-indexed here)
    # reset_term(i,t) = theta * sum_{k=1}^{t} beta^(t-k) * tau[i,k-1]
    #               (1-indexed t,k). In 0-indexed: for output index t0 in [0..T-1],
    #               reset uses tau at indices 0..t0-1.
    tau = torch.zeros(N, T, dtype=torch.float64)
    m = margin.double()

    for _ in range(T + 1):
        # reset_term[i, t0] = theta * sum_{j=0}^{t0-1} beta^(t0-1-j) * tau[i, j]
        reset_term = torch.zeros(N, T, dtype=torch.float64)
        for t0 in range(1, T):
            j = torch.arange(0, t0)
            coeff = theta * (beta ** (t0 - 1 - j).double())          # [t0]
            reset_term[:, t0] = (tau[:, :t0] * coeff.unsqueeze(0)).sum(dim=1)
        total_bound = B_input + reset_term
        tau_next = (m <= total_bound).double()                      # NON-strict, matches proof
        if torch.equal(tau_next, tau):
            tau = tau_next
            break
        tau = tau_next
    return tau


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_samples(dataset, n_samples, fake=False, seed=0):
    in_dim = 3072 if dataset == "cifar10" else 784
    if fake:
        g = torch.Generator().manual_seed(seed)
        return torch.rand(n_samples, in_dim, generator=g), in_dim
    from torchvision import datasets, transforms
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])
    DS = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST,
          "cifar10": datasets.CIFAR10}[dataset]
    ds = DS("./data", train=False, download=True, transform=tfm)
    xs = torch.stack([ds[i][0].view(-1) for i in range(n_samples)])
    return xs, in_dim


# ----------------------------------------------------------------------
# One configuration (dataset, beta, T)
# ----------------------------------------------------------------------
def run_config(dataset, beta, T, args):
    xs, in_dim = load_samples(dataset, args.n_samples, fake=args.fake, seed=args.seed)
    layer = LIFLayer(in_dim, args.n_hidden, beta=beta, theta=args.theta, seed=args.seed)

    viol_total = 0
    safe_fracs = []
    flip_fracs = []

    for idx in range(xs.shape[0]):
        x = xs[idx:idx+1]
        delta = l2_perturbation(x, args.eps, seed=args.seed + idx)
        xp = x + delta

        Vc, sc = layer.run(x,  T)     # [1, N, T]
        Vp, sp = layer.run(xp, T)
        Vc, sc, Vp, sp = Vc[0], sc[0], Vp[0], sp[0]   # [N, T]

        e_true = (sp - sc).abs()                       # [N, T] in {0,1}
        margin = (Vc - layer.theta).abs()              # [N, T]

        tau = least_fixed_point(margin, layer.row_norms(), beta,
                                args.theta, args.eps, T)            # [N, T] float64

        # --- soundness check: no (tau==0 AND e_true==1) ---
        bad = ((tau == 0) & (e_true.double() == 1))
        n_bad = int(bad.sum().item())
        if n_bad > 0:
            ii, tt = torch.where(bad)
            raise AssertionError(
                f"SOUNDNESS VIOLATION ds={dataset} beta={beta} T={T} sample={idx}: "
                f"{n_bad} cases, first at neuron={ii[0].item()} t={tt[0].item()} "
                f"(tau=0 but spike flipped)")
        viol_total += n_bad

        safe_fracs.append(float((1.0 - tau).mean().item()))
        flip_fracs.append(float(e_true.mean().item()))

    return {
        "dataset": dataset, "beta": beta, "T": T,
        "viol": viol_total,
        "safe_frac": float(np.mean(safe_fracs)),
        "flip_frac": float(np.mean(flip_fracs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashion", "cifar10"])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.8, 0.9, 0.99])
    ap.add_argument("--timesteps", nargs="+", type=int, default=[10, 20, 50])
    ap.add_argument("--n-hidden", type=int, default=128)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    print(f"LFP certificate validation | hidden={args.n_hidden} eps={args.eps} "
          f"theta={args.theta} samples={args.n_samples}", flush=True)
    print(f"reset = subtraction | datasets={args.datasets} betas={args.betas} T={args.timesteps}\n",
          flush=True)

    results = []
    for ds in args.datasets:
        for beta in args.betas:
            for T in args.timesteps:
                r = run_config(ds, beta, T, args)
                results.append(r)
                print(f"  {ds:8s} beta={beta:<4} T={T:<3} | "
                      f"viol={r['viol']:<4} | "
                      f"certified-safe={100*r['safe_frac']:5.1f}% | "
                      f"flip-rate={100*r['flip_frac']:4.1f}%", flush=True)

    # --- summary tables ---
    print("\n=== SOUNDNESS ===", flush=True)
    total_viol = sum(r["viol"] for r in results)
    print(f"Total violations across all {len(results)} configs: {total_viol}  "
          f"({'PASS - proof confirmed' if total_viol == 0 else 'FAIL'})", flush=True)

    print("\n=== CERTIFIED-SAFE FRACTION  (rows=beta, cols=T) ===", flush=True)
    for ds in args.datasets:
        print(f"\n[{ds}]", flush=True)
        header = "beta\\T  " + "".join(f"{T:>8}" for T in args.timesteps)
        print(header, flush=True)
        for beta in args.betas:
            row = f"{beta:<6}  "
            for T in args.timesteps:
                r = next(x for x in results if x["dataset"] == ds
                         and x["beta"] == beta and x["T"] == T)
                row += f"{100*r['safe_frac']:7.1f}%"
            print(row, flush=True)

    print("\nALL DONE", flush=True)
    return results


if __name__ == "__main__":
    main()
