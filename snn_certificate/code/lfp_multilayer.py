#!/usr/bin/env python3
"""
lfp_multilayer.py
=================
End-to-end test of the spike-preservation certificate chain on a multi-layer
convolutional SNN under reset-by-subtraction:

  Theorem 1  : per-neuron, per-timestep spike-preservation flags tau (each layer)
  Prop 2     : per-neuron count-change bound  Delta_count(i) <= sum_t tau(i,t)
  Compositional propagation : layer L's flags drive layer L+1's input bound
  Prop 3     : output-margin certificate  M > Delta+ + Delta-  =>  class preserved

WHAT THIS VALIDATES (the two decisive numbers):
  (a) viol == 0 at EVERY spiking layer  -> the COMPOSITION stays sound
      (this is the empirical test of the compositional-soundness link, which is
       derived but not yet independently proven)
  (b) output certified-robust fraction  -> is Prop 3 non-vacuous, or does the
      accumulated looseness collapse it to ~0 across depth?

Architecture (strided convs, no pooling -> linear, certificate-clean):
  Conv1-LIF -> Conv2-LIF -> flatten -> FC1-LIF -> FC-out-LIF (n_classes)
  decision = argmax over output spike counts (Diehl-Cook style).

Reset-by-subtraction everywhere:  V = beta*V + drive - theta*s_prev ; s = 1[V>=theta]
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# surrogate spike
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
spike = SurrogateSpike.apply


# ----------------------------------------------------------------------
# Conv-SNN model.  Records V and s per spiking layer when asked.
# ----------------------------------------------------------------------
class ConvSNN(nn.Module):
    def __init__(self, in_ch, n_classes=10, beta=0.9, theta=1.0, img=28):
        super().__init__()
        self.beta, self.theta = beta, theta
        self.c1 = nn.Conv2d(in_ch, 16, 3, stride=2, padding=1, bias=False)   # img->img/2
        self.c2 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False)      # ->img/4
        s = img // 4
        self.flat_dim = 32 * s * s
        self.fc1 = nn.Linear(self.flat_dim, 128, bias=False)
        self.fc_out = nn.Linear(128, n_classes, bias=False)
        self.sdim = s

    def _lif_step(self, drive, V, s_prev):
        V = self.beta * V + drive - self.theta * s_prev
        s = spike(V, self.theta)
        return V, s

    def forward(self, x, T, record=False):
        B = x.size(0)
        dev = x.device
        # init membranes
        V1 = V2 = Vf = Vo = None
        s1p = s2p = sfp = sop = None
        out_count = 0.0
        rec = {k: [] for k in ["V1","s1","V2","s2","Vf","sf","Vo","so"]} if record else None
        for t in range(T):
            d1 = self.c1(x)
            if V1 is None:
                V1 = torch.zeros_like(d1); s1p = torch.zeros_like(d1)
            V1, s1 = self._lif_step(d1, V1, s1p); s1p = s1
            d2 = self.c2(s1)
            if V2 is None:
                V2 = torch.zeros_like(d2); s2p = torch.zeros_like(d2)
            V2, s2 = self._lif_step(d2, V2, s2p); s2p = s2
            f = s2.flatten(1)
            df = self.fc1(f)
            if Vf is None:
                Vf = torch.zeros_like(df); sfp = torch.zeros_like(df)
            Vf, sf = self._lif_step(df, Vf, sfp); sfp = sf
            do = self.fc_out(sf)
            if Vo is None:
                Vo = torch.zeros_like(do); sop = torch.zeros_like(do)
            Vo, so = self._lif_step(do, Vo, sop); sop = so
            out_count = out_count + so
            if record:
                rec["V1"].append(V1); rec["s1"].append(s1)
                rec["V2"].append(V2); rec["s2"].append(s2)
                rec["Vf"].append(Vf); rec["sf"].append(sf)
                rec["Vo"].append(Vo); rec["so"].append(so)
        if record:
            for k in rec: rec[k] = torch.stack(rec[k], dim=-1)  # [...,T]
            return out_count, rec
        return out_count


# ----------------------------------------------------------------------
# discounted causal sums (0-indexed time)
#   reset variant:  y[t] = sum_{j=0}^{t-1} beta^(t-1-j) x[j]          (prev spikes)
#   input variant:  y[t] = sum_{j=0}^{t}   beta^(t-j)   x[j]          (current drive)
# ----------------------------------------------------------------------
def discounted_reset(x, beta):           # x: [..., T] -> [..., T]
    T = x.shape[-1]
    y = torch.zeros_like(x)
    for t in range(1, T):
        j = torch.arange(0, t, device=x.device)
        coeff = beta ** (t - 1 - j).to(x.dtype)
        y[..., t] = (x[..., :t] * coeff).sum(-1)
    return y

def discounted_input(x, beta):           # x: [..., T] -> [..., T]
    T = x.shape[-1]
    y = torch.zeros_like(x)
    for t in range(0, T):
        j = torch.arange(0, t + 1, device=x.device)
        coeff = beta ** (t - j).to(x.dtype)
        y[..., t] = (x[..., :t + 1] * coeff).sum(-1)
    return y


# ----------------------------------------------------------------------
# Least fixed point given a per-neuron-per-timestep input bound B_in[N,T]
# margin[N,T], returns tau[N,T] in {0,1}.  (generic; works for any layer)
# ----------------------------------------------------------------------
def least_fixed_point(margin, B_in, beta, theta):
    m = margin
    tau = torch.zeros_like(m)
    T = m.shape[-1]
    for _ in range(T + 1):
        reset_term = theta * discounted_reset(tau, beta)
        tau_next = (m <= B_in + reset_term).to(m.dtype)
        if torch.equal(tau_next, tau):
            tau = tau_next; break
        tau = tau_next
    return tau


# ----------------------------------------------------------------------
# l2 perturbations (exact ||delta||_2 = eps)
# ----------------------------------------------------------------------
def l2_gaussian(x, eps, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    n = torch.randn(x.shape, generator=g, device=x.device)
    fl = n.flatten(1); fl = fl / fl.norm(dim=1, keepdim=True).clamp_min(1e-12) * eps
    return fl.view_as(x)

def l2_fgsm(model, x, y, eps, T):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x, T), y)
    grad = torch.autograd.grad(loss, x)[0]
    fl = grad.flatten(1); fl = fl / fl.norm(dim=1, keepdim=True).clamp_min(1e-12) * eps
    return (x + fl.view_as(x)).detach()


# ----------------------------------------------------------------------
# Per-layer input-perturbation bounds
# ----------------------------------------------------------------------
def conv_rownorm(weight):                 # [out,in,kh,kw] -> per-out-channel L2 norm
    return weight.flatten(1).norm(dim=1)  # [out]

def propagate_conv(tau_up, weight_abs, stride, padding, beta):
    """A_up[j,t] = (|W| conv tau_up[:,:,:,t])(j); then B_in[j,t]=sum_k beta^(t-k)A_up[k].
       tau_up: [Cin,H,W,T] -> returns B_in [Cout,Hout,Wout,T]."""
    Cin, H, Wd, T = tau_up.shape
    # move T to batch for conv
    x = tau_up.permute(3, 0, 1, 2)                       # [T,Cin,H,W]
    A = F.conv2d(x, weight_abs, stride=stride, padding=padding)  # [T,Cout,Ho,Wo]
    A = A.permute(1, 2, 3, 0)                            # [Cout,Ho,Wo,T]
    B_in = discounted_input(A, beta)
    return B_in

def propagate_fc(tau_up_flat, weight_abs, beta):
    """tau_up_flat: [Nin,T]; weight_abs: [Nout,Nin]; returns B_in [Nout,T]."""
    A = weight_abs @ tau_up_flat                         # [Nout,T]
    return discounted_input(A, beta)


# ----------------------------------------------------------------------
# Full certificate chain for ONE sample
# ----------------------------------------------------------------------
@torch.no_grad()
def certify_sample(model, x, xp, T, eps, theta, beta):
    # clean + perturbed recordings
    _, rc = model(x,  T, record=True)
    _, rp = model(xp, T, record=True)

    out = {}
    # ---- Layer 1 (conv1): input bound from eps ----
    V1 = rc["V1"][0]; s1c = rc["s1"][0]; s1p = rp["s1"][0]    # [16,H,W,T]
    Co, H, Wd, _ = V1.shape
    rn1 = conv_rownorm(model.c1.weight)                       # [16]
    tfac = ((1 - beta ** torch.arange(1, T + 1, device=x.device).double())
            / (1 - beta))                                     # [T]
    B1 = eps * rn1.double().view(Co, 1, 1, 1) * tfac.view(1, 1, 1, T)  # [16,1,1,T]->bc
    B1 = B1.expand(Co, H, Wd, T)
    m1 = (V1.double() - theta).abs()
    tau1 = least_fixed_point(m1, B1, beta, theta)
    e1 = (s1p - s1c).abs().double()
    viol1 = int(((tau1 == 0) & (e1 == 1)).sum().item())
    out["L1"] = {"viol": viol1, "safe_frac": float((1 - tau1).mean().item())}

    # ---- Layer 2 (conv2): input bound from tau1 propagation ----
    V2 = rc["V2"][0]; s2c = rc["s2"][0]; s2p = rp["s2"][0]
    B2 = propagate_conv(tau1, model.c2.weight.abs().double(), 2, 1, beta)
    m2 = (V2.double() - theta).abs()
    tau2 = least_fixed_point(m2, B2, beta, theta)
    e2 = (s2p - s2c).abs().double()
    viol2 = int(((tau2 == 0) & (e2 == 1)).sum().item())
    out["L2"] = {"viol": viol2, "safe_frac": float((1 - tau2).mean().item())}

    # ---- Layer 3 (fc1): flatten tau2, propagate through fc1 ----
    Vf = rc["Vf"][0]; sfc = rc["sf"][0]; sfp = rp["sf"][0]    # [128,T]
    tau2_flat = tau2.reshape(-1, T)                           # [32*s*s, T]
    Bf = propagate_fc(tau2_flat, model.fc1.weight.abs().double(), beta)
    mf = (Vf.double() - theta).abs()
    tauf = least_fixed_point(mf, Bf, beta, theta)
    ef = (sfp - sfc).abs().double()
    violf = int(((tauf == 0) & (ef == 1)).sum().item())
    out["L3"] = {"viol": violf, "safe_frac": float((1 - tauf).mean().item())}

    # ---- Layer 4 (fc_out): propagate through fc_out ----
    Vo = rc["Vo"][0]; soc = rc["so"][0]; sop = rp["so"][0]    # [n_classes,T]
    Bo = propagate_fc(tauf, model.fc_out.weight.abs().double(), beta)
    mo = (Vo.double() - theta).abs()
    tauo = least_fixed_point(mo, Bo, beta, theta)
    eo = (sop - soc).abs().double()
    violo = int(((tauo == 0) & (eo == 1)).sum().item())
    out["L4"] = {"viol": violo, "safe_frac": float((1 - tauo).mean().item())}

    # ---- Prop 2: count bounds at output ----
    delta_count = tauo.sum(-1)                               # [n_classes]
    # ---- Prop 3: output-margin certificate ----
    Cc = soc.sum(-1).double()                                # clean counts [n_classes]
    top = int(torch.argmax(Cc).item())
    others = Cc.clone(); others[top] = -1e9
    runner = int(torch.argmax(others).item())
    M = float((Cc[top] - Cc[runner]).item())
    delta_plus = float(tauo[top].sum().item())               # top can lose
    delta_minus = float(max(tauo[c].sum().item() for c in range(Cc.shape[0]) if c != top))
    certified_robust = bool(M > delta_plus + delta_minus)

    # ---- actual robustness (referee) ----
    Cp = sop.sum(-1).double()
    actual_robust = bool(int(torch.argmax(Cp).item()) == top)
    # output soundness: certified_robust must imply actual_robust
    out_viol = int(certified_robust and (not actual_robust))

    out["output"] = {
        "clean_top": top, "margin_M": M,
        "delta_plus": delta_plus, "delta_minus": delta_minus,
        "certified_robust": certified_robust,
        "actual_robust": actual_robust,
        "output_soundness_viol": out_viol,
    }
    return out


# ----------------------------------------------------------------------
# data / training
# ----------------------------------------------------------------------
def get_data(dataset, batch, fake):
    in_ch = 3 if dataset == "cifar10" else 1
    img = 32 if dataset == "cifar10" else 28
    if fake:
        Xtr = torch.rand(256, in_ch, img, img); Ytr = torch.randint(0, 10, (256,))
        Xte = torch.rand(64, in_ch, img, img);  Yte = torch.randint(0, 10, (64,))
        tr = torch.utils.data.TensorDataset(Xtr, Ytr)
        te = torch.utils.data.TensorDataset(Xte, Yte)
        return (torch.utils.data.DataLoader(tr, batch, shuffle=True),
                torch.utils.data.DataLoader(te, batch, shuffle=False), in_ch, img)
    from torchvision import datasets, transforms
    if dataset == "cifar10":
        tfm = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,)*3, (0.5,)*3)])
        DS = datasets.CIFAR10
    else:
        tfm = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))])
        DS = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}[dataset]
    tr = DS("./data", train=True, download=True, transform=tfm)
    te = DS("./data", train=False, download=True, transform=tfm)
    return (torch.utils.data.DataLoader(tr, batch, shuffle=True),
            torch.utils.data.DataLoader(te, batch, shuffle=False), in_ch, img)

def train(model, loader, dev, T, epochs):
    opt = torch.optim.Adam(model.parameters(), 1e-3); model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = F.cross_entropy(model(xb, T), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"      epoch {ep+1}/{epochs}", flush=True)
    model.eval()

@torch.no_grad()
def accuracy(model, loader, dev, T, maxb=20):
    c = n = 0
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(dev), yb.to(dev)
        c += (model(xb, T).argmax(1) == yb).sum().item(); n += yb.size(0)
        if i + 1 >= maxb: break
    return c / max(n, 1)


# ----------------------------------------------------------------------
# run one (dataset, T)
# ----------------------------------------------------------------------
def run(dataset, T, args, dev):
    tr, te, in_ch, img = get_data(dataset, args.batch, args.fake)
    model = ConvSNN(in_ch, beta=args.beta, theta=args.theta, img=img).to(dev)
    print(f"  training {dataset} T={T} ...", flush=True)
    train(model, tr, dev, T, args.epochs)
    acc = accuracy(model, te, dev, T)
    ckpt = f"{args.outdir}/ckpt_{dataset}_T{T}.pt"
    torch.save(model.state_dict(), ckpt)

    xb, yb = next(iter(te)); xb, yb = xb.to(dev)[:args.n_samples], yb.to(dev)[:args.n_samples]

    agg = {f"L{l}": {"viol": 0, "safe": []} for l in range(1, 5)}
    cert_robust = 0; actual_robust = 0; out_viol = 0; margins = []
    for idx in range(xb.size(0)):
        x = xb[idx:idx+1]
        if args.attack == "fgsm":
            delta = l2_fgsm(model, x, yb[idx:idx+1], args.eps, T) - x
            xp = x + delta
        else:
            xp = x + l2_gaussian(x, args.eps, seed=args.seed + idx)
        r = certify_sample(model, x, xp, T, args.eps, args.theta, args.beta)
        for l in range(1, 5):
            agg[f"L{l}"]["viol"] += r[f"L{l}"]["viol"]
            agg[f"L{l}"]["safe"].append(r[f"L{l}"]["safe_frac"])
        o = r["output"]
        cert_robust += int(o["certified_robust"])
        actual_robust += int(o["actual_robust"])
        out_viol += o["output_soundness_viol"]
        margins.append(o["margin_M"])

    n = xb.size(0)
    res = {
        "dataset": dataset, "T": T, "clean_acc": acc, "n_samples": n,
        "attack": args.attack, "eps": args.eps, "beta": args.beta,
        "layers": {l: {"viol": agg[l]["viol"],
                       "safe_frac": float(np.mean(agg[l]["safe"]))} for l in agg},
        "output": {
            "certified_robust_frac": cert_robust / n,
            "actual_robust_frac": actual_robust / n,
            "output_soundness_viol": out_viol,
            "mean_margin": float(np.mean(margins)),
        },
        "checkpoint": ckpt,
    }
    # print
    print(f"  [{dataset} T={T}] acc={acc:.3f}", flush=True)
    for l in range(1, 5):
        print(f"    {l}: viol={res['layers'][f'L{l}']['viol']:<3} "
              f"safe={100*res['layers'][f'L{l}']['safe_frac']:5.1f}%", flush=True)
    print(f"    OUTPUT: certified-robust={100*res['output']['certified_robust_frac']:5.1f}%  "
          f"actual-robust={100*res['output']['actual_robust_frac']:5.1f}%  "
          f"out-viol={out_viol}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashion", "cifar10"])
    ap.add_argument("--timesteps", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--attack", choices=["gaussian", "fgsm"], default="gaussian")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="lfp_ml_out")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"multilayer LFP chain | device={dev} attack={args.attack} eps={args.eps} "
          f"beta={args.beta} epochs={args.epochs} samples={args.n_samples}", flush=True)

    results = []
    for ds in args.datasets:
        for T in args.timesteps:
            results.append(run(ds, T, args, dev))

    # summary
    print("\n=== SOUNDNESS (viol must be 0 at every layer AND output) ===", flush=True)
    tot = sum(r["layers"][f"L{l}"]["viol"] for r in results for l in range(1, 5))
    tot += sum(r["output"]["output_soundness_viol"] for r in results)
    print(f"Total violations across all layers+output: {tot}  "
          f"({'PASS' if tot == 0 else 'FAIL'})", flush=True)

    print("\n=== OUTPUT CERTIFIED-ROBUST FRACTION (Prop 3) ===", flush=True)
    print(f"{'dataset':10}{'T':>4}{'acc':>8}{'cert-rob':>10}{'act-rob':>9}", flush=True)
    for r in results:
        print(f"{r['dataset']:10}{r['T']:>4}{r['clean_acc']:>8.3f}"
              f"{100*r['output']['certified_robust_frac']:>9.1f}%"
              f"{100*r['output']['actual_robust_frac']:>8.1f}%", flush=True)

    with open(f"{args.outdir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {args.outdir}/results.json and checkpoints", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
