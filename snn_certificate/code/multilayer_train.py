#!/usr/bin/env python3
"""
multilayer_train.py  --  TRAINING-FIXED multilayer conv-SNN (reset-by-subtraction)

This file's ONE job: make the deep conv-SNN actually TRAIN to real accuracy.
(The certificate is NOT run here -- we add it back only after training is confirmed.)

Fixes vs the previous version that sat at ~10%:
  1. bias=True on conv+fc           -> neurons can reach a useful firing regime
  2. batch-norm after each conv     -> stabilizes the pre-spike drive across depth
  3. non-spiking readout on the      -> gradient no longer dies through a spiking
     LAST hidden layer's spike-rate     output layer; trains like a normal net
  4. spike-rate output (mean over T) -> bounded, well-scaled logits
  5. lighter first stride            -> keeps spatial info for early features

FAST SANITY GATE (run this FIRST, ~2 min on any GPU, before the full job):
    python multilayer_train.py --sanity
  It trains MNIST T=8 for 2 epochs and PRINTS the accuracy. If acc > 0.5 the
  architecture trains and you can launch the full run. If not, STOP and tell me
  -- do NOT waste cluster hours on the full grid.

Full run:
    python multilayer_train.py --datasets mnist fashion cifar10 --timesteps 4 8 --epochs 8
"""

import argparse, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


# ---- surrogate spike (kept identical to the certified model so dynamics match) ----
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, theta):
        ctx.save_for_backward(v); ctx.theta = theta
        return (v >= theta).float()
    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        return g * torch.clamp(1 - (v - ctx.theta).abs(), min=0.0), None
spike = SurrogateSpike.apply


class ConvSNN(nn.Module):
    """
    Spiking forward pass at every layer (reset-by-subtraction) is preserved so the
    certificate can still be applied later. Trainability comes from BN + bias + a
    non-spiking rate readout. Decision = argmax of the readout (NOT spike-count),
    which is what makes it train; the spiking layers underneath are unchanged.
    """
    def __init__(self, in_ch, n_classes=10, beta=0.9, theta=1.0, img=28):
        super().__init__()
        self.beta, self.theta = beta, theta
        self.c1 = nn.Conv2d(in_ch, 32, 3, stride=1, padding=1, bias=True)   # keep res
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=True)      # /2
        self.bn2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=True)      # /4
        self.bn3 = nn.BatchNorm2d(64)
        s = img // 4
        self.fc1 = nn.Linear(64 * s * s, 256, bias=True)
        self.readout = nn.Linear(256, n_classes)   # NON-spiking readout (trainable)

    def _lif(self, drive, V, sprev):
        V = self.beta * V + drive - self.theta * sprev
        return V, spike(V, self.theta)

    def forward(self, x, T):
        V1 = V2 = V3 = Vf = None
        s1p = s2p = s3p = sfp = None
        rate = 0.0
        for _ in range(T):
            d = self.bn1(self.c1(x))
            if V1 is None: V1 = torch.zeros_like(d); s1p = torch.zeros_like(d)
            V1, s1 = self._lif(d, V1, s1p); s1p = s1
            d = self.bn2(self.c2(s1))
            if V2 is None: V2 = torch.zeros_like(d); s2p = torch.zeros_like(d)
            V2, s2 = self._lif(d, V2, s2p); s2p = s2
            d = self.bn3(self.c3(s2))
            if V3 is None: V3 = torch.zeros_like(d); s3p = torch.zeros_like(d)
            V3, s3 = self._lif(d, V3, s3p); s3p = s3
            d = self.fc1(s3.flatten(1))
            if Vf is None: Vf = torch.zeros_like(d); sfp = torch.zeros_like(d)
            Vf, sf = self._lif(d, Vf, sfp); sfp = sf
            rate = rate + sf
        return self.readout(rate / T)     # bounded, well-scaled logits


def get_data(ds, batch, fake=False):
    inch = 3 if ds == "cifar10" else 1
    img = 32 if ds == "cifar10" else 28
    if fake:
        tr = torch.utils.data.TensorDataset(torch.rand(512, inch, img, img), torch.randint(0, 10, (512,)))
        te = torch.utils.data.TensorDataset(torch.rand(128, inch, img, img), torch.randint(0, 10, (128,)))
        return (torch.utils.data.DataLoader(tr, batch, shuffle=True),
                torch.utils.data.DataLoader(te, batch, shuffle=False), inch, img)
    from torchvision import datasets, transforms
    if ds == "cifar10":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
        DS = datasets.CIFAR10
    else:
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        DS = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}[ds]
    tr = DS("./data", train=True, download=True, transform=tfm)
    te = DS("./data", train=False, download=True, transform=tfm)
    return (torch.utils.data.DataLoader(tr, batch, shuffle=True),
            torch.utils.data.DataLoader(te, batch, shuffle=False), inch, img)


def train(model, loader, dev, T, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    model.train()
    for ep in range(epochs):
        tot = corr = seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            logits = model(xb, T)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * yb.size(0)
            corr += (logits.argmax(1) == yb).sum().item(); seen += yb.size(0)
        sched.step()
        print(f"      epoch {ep+1}/{epochs}  loss={tot/seen:.3f}  train_acc={corr/seen:.3f}", flush=True)
    model.eval()


@torch.no_grad()
def test_acc(model, loader, dev, T, maxb=40):
    c = n = 0
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(dev), yb.to(dev)
        c += (model(xb, T).argmax(1) == yb).sum().item(); n += yb.size(0)
        if i + 1 >= maxb: break
    return c / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashion", "cifar10"])
    ap.add_argument("--timesteps", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--sanity", action="store_true",
                    help="2-min MNIST T=8 gate; prints acc and exits")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    if args.sanity:
        print(f"[SANITY GATE] device={dev}  MNIST T=8, 2 epochs", flush=True)
        tr, te, inch, img = get_data("mnist", args.batch, fake=args.fake)
        model = ConvSNN(inch, beta=args.beta, theta=args.theta, img=img).to(dev)
        train(model, tr, dev, 8, 2, args.lr)
        acc = test_acc(model, te, dev, 8)
        verdict = "PASS - architecture trains, launch the full run" if acc > 0.5 \
                  else "FAIL - do NOT launch full run; report this acc back"
        print(f"\n[SANITY GATE]  MNIST T=8 test_acc = {acc:.3f}   -> {verdict}", flush=True)
        return

    results = []
    for ds in args.datasets:
        for T in args.timesteps:
            print(f"  training {ds} T={T} ...", flush=True)
            tr, te, inch, img = get_data(ds, args.batch, fake=args.fake)
            model = ConvSNN(inch, beta=args.beta, theta=args.theta, img=img).to(dev)
            train(model, tr, dev, T, args.epochs, args.lr)
            acc = test_acc(model, te, dev, T)
            ckpt = f"ckpt_{ds}_T{T}.pt"
            torch.save(model.state_dict(), ckpt)
            results.append({"ds": ds, "T": T, "test_acc": round(acc, 4), "ckpt": ckpt})
            print(f"  [{ds} T={T}] TEST ACC = {acc:.4f}  (saved {ckpt})\n", flush=True)

    print("==== TRAINING SUMMARY (copy back) ====")
    print(json.dumps(results, indent=2))
    print("==== END ====")
    with open("train_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
