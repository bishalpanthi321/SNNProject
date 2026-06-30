#!/usr/bin/env python3
"""
deep_core.py
============
Per-neuron spike-pattern certificate sweep for feedforward LIF nets.

For each (beta, timesteps) it trains a small SNN, then for each epsilon it
runs a clean vs L2-perturbed forward pass and evaluates a PER-NEURON,
PER-TIMESTEP certificate on the HIDDEN layer:

  B(i,t,eps) = eps * ||row_i(W1)||_2 * (1 - beta**t)/(1 - beta)      (tight bound)
  tau(i,t)   = 1[ |V_hid(i,t) - theta| < B(i,t,eps) ]               (could-flip flag)
  flip(i,t)  = 1[ spk_hid_clean(i,t) != spk_hid_pert(i,t) ]         (did flip)

Reported per (beta, T, eps):
  vuln%      = mean tau                       (fraction flagged vulnerable)
  flip%      = mean flip                      (fraction that actually flipped)
  viol       = #(flip AND NOT tau)            (CERTIFICATE VIOLATIONS; want 0)
  precision  = #(flip AND tau)/#(tau)         (tightness: flagged that flipped)
  d_out      = global output-spike change     (descriptive only)

No file output: everything is printed to stdout for copy-paste.
"""

import argparse
import torch
import torch.nn as nn
import snntorch as snn


DATASET_INFO = {
    "mnist":         {"input_size": 784,  "n_classes": 10, "tv": "MNIST",
                      "norm": ((0.5,), (0.5,))},
    "fashion_mnist": {"input_size": 784,  "n_classes": 10, "tv": "FashionMNIST",
                      "norm": ((0.5,), (0.5,))},
    "cifar10":       {"input_size": 3072, "n_classes": 10, "tv": "CIFAR10",
                      "norm": ((0.5,) * 3, (0.5,) * 3)},
}


class LIFNet(nn.Module):
    def __init__(self, input_size, hidden, output, beta, threshold=1.0,
                 reset="subtract"):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset)
        self.fc2 = nn.Linear(hidden, output)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset)
        self.beta = beta
        self.threshold = threshold

    def forward(self, x, steps):
        x = x.view(x.size(0), -1)
        mem1 = mem2 = None
        s1, m1, s2, m2 = [], [], [], []
        for _ in range(steps):
            cur1 = self.fc1(x)
            if mem1 is None:
                mem1 = torch.zeros_like(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            if mem2 is None:
                mem2 = torch.zeros_like(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            s1.append(spk1); m1.append(mem1); s2.append(spk2); m2.append(mem2)
        return {"spk_hid": torch.stack(s1), "mem_hid": torch.stack(m1),
                "spk_out": torch.stack(s2), "mem_out": torch.stack(m2)}

    def logits(self, x, steps):
        return self.forward(x, steps)["spk_out"].sum(0)


def l2_perturb(x, eps):
    noise = torch.randn_like(x)
    flat = noise.view(noise.size(0), -1)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return flat.view_as(x) * eps


def hidden_rownorms(model):
    return torch.linalg.norm(model.fc1.weight.detach(), dim=1)   # [hidden]


def time_factor(beta, T, device):
    s = torch.arange(1, T + 1, device=device, dtype=torch.float32)
    return (1.0 - beta ** s) / (1.0 - beta)                      # [T]


def get_loaders(dataset, batch_size, data_dir, subset, fake_data):
    info = DATASET_INFO[dataset]
    if fake_data:
        n = subset if subset else 256
        tr = torch.utils.data.TensorDataset(
            torch.rand(n, info["input_size"]),
            torch.randint(0, info["n_classes"], (n,)))
        te = torch.utils.data.TensorDataset(
            torch.rand(min(n, 256), info["input_size"]),
            torch.randint(0, info["n_classes"], (min(n, 256),)))
        return (torch.utils.data.DataLoader(tr, batch_size, shuffle=True),
                torch.utils.data.DataLoader(te, batch_size, shuffle=False))

    from torchvision import datasets, transforms
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(*info["norm"])])
    DS = getattr(datasets, info["tv"])
    train = DS(data_dir, train=True, download=True, transform=tfm)
    test = DS(data_dir, train=False, download=True, transform=tfm)
    if subset:
        train = torch.utils.data.Subset(train, list(range(min(subset, len(train)))))
    return (torch.utils.data.DataLoader(train, batch_size, shuffle=True),
            torch.utils.data.DataLoader(test, batch_size, shuffle=False))


def train_model(model, loader, device, steps, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(model.logits(x, steps), y)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


@torch.no_grad()
def evaluate(model, loader, device, steps, max_batches=20):
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        correct += (model.logits(x, steps).argmax(1) == y).sum().item()
        total += y.size(0)
        if i + 1 >= max_batches:
            break
    return correct / max(total, 1)


def single_run(dataset, beta, steps, args, device):
    info = DATASET_INFO[dataset]
    tl, vl = get_loaders(dataset, args.batch_size, args.data_dir,
                         args.subset, args.fake_data)
    model = LIFNet(info["input_size"], args.hidden, info["n_classes"],
                   beta=beta, reset=args.reset).to(device)
    train_model(model, tl, device, steps, args.epochs)
    acc = evaluate(model, vl, device, steps)

    x, _ = next(iter(vl))
    x = x.to(device)
    with torch.no_grad():
        clean = model(x, steps)
    mem_hid = clean["mem_hid"]                      # [T,B,H]
    spk_hid = clean["spk_hid"]
    spk_out = clean["spk_out"]
    theta = model.threshold

    rownorm = hidden_rownorms(model).to(device)     # [H]
    factor = time_factor(beta, steps, device)       # [T]
    dist = torch.abs(mem_hid - theta)               # [T,B,H]

    rows = []
    for eps in args.eps:
        B = eps * factor[:, None] * rownorm[None, :]            # [T,H]
        tau = dist < B[:, None, :]                              # [T,B,H] bool
        with torch.no_grad():
            pert = model(x + l2_perturb(x, eps), steps)
        flip = spk_hid != pert["spk_hid"]                       # [T,B,H] bool
        d_out = (spk_out != pert["spk_out"]).float().mean().item()

        flagged = tau.sum().item()
        flipped = flip.sum().item()
        viol = (flip & ~tau).sum().item()
        hit = (flip & tau).sum().item()
        vuln = tau.float().mean().item()
        flipr = flip.float().mean().item()
        prec = hit / flagged if flagged > 0 else 0.0

        rows.append({"eps": eps, "vuln": vuln, "flip": flipr, "viol": viol,
                     "prec": prec, "d_out": d_out, "flipped": flipped})
        print(f"    eps={eps:<7} vuln={100*vuln:6.2f}%  flip={100*flipr:6.3f}%  "
              f"viol={viol:<6} prec={prec:6.3f}  d_out={d_out:.4f}", flush=True)

    # empirical sound radius: largest eps with zero violations
    sound_eps = 0.0
    for r in rows:
        if r["viol"] == 0:
            sound_eps = max(sound_eps, r["eps"])
    viol_total = sum(r["viol"] for r in rows)
    return {"beta": beta, "T": steps, "acc": acc, "rows": rows,
            "sound_eps": sound_eps, "viol_total": viol_total}


def run_experiment(dataset, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"dataset={dataset} device={device} betas={args.betas} "
          f"T={args.timesteps} epochs={args.epochs} reset={args.reset} "
          f"eps={args.eps}", flush=True)

    runs = []
    total = len(args.betas) * len(args.timesteps)
    k = 0
    for beta in args.betas:
        for steps in args.timesteps:
            k += 1
            print(f"\n[{k}/{total}] beta={beta} T={steps}", flush=True)
            r = single_run(dataset, beta, steps, args, device)
            print(f"    -> acc={r['acc']:.4f}  viol_total={r['viol_total']}  "
                  f"sound_eps={r['sound_eps']}", flush=True)
            runs.append(r)

    # ---- final paste-friendly summary, probed at the middle epsilon ----
    mid = args.eps[len(args.eps) // 2]
    print("\n\n======== PASTE THIS TABLE ========", flush=True)
    print(f"# dataset={dataset}  probe_eps={mid}  reset={args.reset}  "
          f"epochs={args.epochs}", flush=True)
    print(f"{'beta':>5} {'T':>3} {'acc':>6} {'viol_tot':>8} {'sound_eps':>9} "
          f"{'vuln%':>7} {'flip%':>8} {'prec':>6}", flush=True)
    for r in runs:
        pr = next(x for x in r["rows"] if x["eps"] == mid)
        print(f"{r['beta']:>5} {r['T']:>3} {r['acc']:>6.4f} "
              f"{r['viol_total']:>8} {r['sound_eps']:>9} "
              f"{100*pr['vuln']:>7.2f} {100*pr['flip']:>8.3f} {pr['prec']:>6.3f}",
              flush=True)
    print("==================================", flush=True)
    print("\nALL DONE", flush=True)


def build_argparser(dataset):
    ap = argparse.ArgumentParser(description=f"per-neuron SNN certificate: {dataset}")
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    ap.add_argument("--timesteps", type=int, nargs="+",
                    default=[5, 10, 15, 20, 25, 30])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    ap.add_argument("--reset", choices=["subtract", "zero", "none"],
                    default="subtract")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--fake-data", action="store_true")
    return ap
