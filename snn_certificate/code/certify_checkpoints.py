#!/usr/bin/env python3
"""
certify_checkpoints.py
======================
Loads the TRAINED checkpoints from multilayer_train.py and runs the full
spike-preservation certificate chain on them. NO training here -> fast (minutes).

Chain:  Theorem 1 (per-layer flags) -> Prop 2 (count bound) ->
        compositional propagation L1->L2->L3->L4 -> Prop 3 (output margin).

CORRECTNESS NOTES (these matter for soundness):
  * The trained model has BatchNorm after each conv. BN is a per-channel affine
    map y = gamma*(x-mu)/sqrt(var+eps) + bias. At eval time it is a FIXED linear
    rescale, so we FOLD it into the conv weights to get the effective weight whose
    row-norm drives B_input. Folding factor per output channel c: gamma_c/sqrt(var_c+eps).
  * The readout is NON-spiking (Linear on the last hidden layer's spike-rate).
    The decision is argmax of readout(rate/T). Prop 3 is therefore stated on the
    READOUT logits: a class flips only if the logit margin is eroded by the
    certified change in the last hidden layer's spike counts, propagated through
    the readout weights. We compute that erosion soundly from the L4 flags.
  * viol is checked at every spiking layer (L1..L4). It MUST be 0.

Run:
    python certify_checkpoints.py --eps 0.1 --n-samples 30
"""

import argparse, json, glob, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ---- must match the trained model EXACTLY ----
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
    def __init__(self, in_ch, n_classes=10, beta=0.9, theta=1.0, img=28):
        super().__init__()
        self.beta, self.theta = beta, theta
        self.c1 = nn.Conv2d(in_ch, 32, 3, 1, 1, bias=True); self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3, 2, 1, bias=True);   self.bn2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 64, 3, 2, 1, bias=True);   self.bn3 = nn.BatchNorm2d(64)
        s = img // 4
        self.fc1 = nn.Linear(64*s*s, 256, bias=True)
        self.readout = nn.Linear(256, n_classes)
    def _lif(self, d, V, sp):
        V = self.beta*V + d - self.theta*sp
        return V, spike(V, self.theta)
    @torch.no_grad()
    def run(self, x, T):
        """forward that RECORDS V,s at every spiking layer."""
        V1=V2=V3=Vf=None; s1p=s2p=s3p=sfp=None
        rec={k:[] for k in ["V1","s1","V2","s2","V3","s3","Vf","sf"]}
        for _ in range(T):
            d=self.bn1(self.c1(x))
            if V1 is None: V1=torch.zeros_like(d); s1p=torch.zeros_like(d)
            V1,s1=self._lif(d,V1,s1p); s1p=s1
            d=self.bn2(self.c2(s1))
            if V2 is None: V2=torch.zeros_like(d); s2p=torch.zeros_like(d)
            V2,s2=self._lif(d,V2,s2p); s2p=s2
            d=self.bn3(self.c3(s2))
            if V3 is None: V3=torch.zeros_like(d); s3p=torch.zeros_like(d)
            V3,s3=self._lif(d,V3,s3p); s3p=s3
            d=self.fc1(s3.flatten(1))
            if Vf is None: Vf=torch.zeros_like(d); sfp=torch.zeros_like(d)
            Vf,sf=self._lif(d,Vf,sfp); sfp=sf
            for k,v in [("V1",V1),("s1",s1),("V2",V2),("s2",s2),
                        ("V3",V3),("s3",s3),("Vf",Vf),("sf",sf)]:
                rec[k].append(v)
        for k in rec: rec[k]=torch.stack(rec[k],-1)   # [...,T]
        return rec

    # ---- BN-folded effective weights (per-output-channel rescale folded in) ----
    def eff_conv(self, conv, bn):
        scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps))    # [out]
        W = conv.weight * scale.view(-1,1,1,1)                        # fold into kernel
        return W
    def row_norm_conv(self, conv, bn):
        W = self.eff_conv(conv, bn)
        return W.flatten(1).norm(dim=1)                              # [out]


# ---- discounted causal sums (validated conventions) ----
def disc_reset(x, beta):
    T=x.shape[-1]; y=torch.zeros_like(x)
    for t in range(1,T):
        j=torch.arange(0,t,device=x.device)
        y[...,t]=(x[...,:t]*(beta**(t-1-j).to(x.dtype))).sum(-1)
    return y
def disc_input(x, beta):
    T=x.shape[-1]; y=torch.zeros_like(x)
    for t in range(T):
        j=torch.arange(0,t+1,device=x.device)
        y[...,t]=(x[...,:t+1]*(beta**(t-j).to(x.dtype))).sum(-1)
    return y
def lfp(margin, B_in, beta, theta):
    m=margin; tau=torch.zeros_like(m); T=m.shape[-1]
    for _ in range(T+1):
        tn=(m <= B_in + theta*disc_reset(tau,beta)).to(m.dtype)
        if torch.equal(tn,tau): tau=tn; break
        tau=tn
    return tau

def l2_gauss(x, eps, seed=0):
    g=torch.Generator(device=x.device).manual_seed(seed)
    n=torch.randn(x.shape,generator=g,device=x.device); fl=n.flatten(1)
    return (fl/fl.norm(dim=1,keepdim=True).clamp_min(1e-12)*eps).view_as(x)

# propagation: upstream flags -> downstream input bound (BN-folded abs weights)
def prop_conv(tau_up, w_abs, stride, pad, beta):
    A=F.conv2d(tau_up.permute(3,0,1,2), w_abs, stride=stride, padding=pad).permute(1,2,3,0)
    return disc_input(A, beta)
def prop_fc(tau_up_flat, w_abs, beta):
    return disc_input(w_abs @ tau_up_flat, beta)


@torch.no_grad()
def certify(model, x, xp, T, eps, theta, beta):
    rc=model.run(x,T); rp=model.run(xp,T)
    res={}; dev=x.device
    tfac=((1-beta**torch.arange(1,T+1,device=dev).double())/(1-beta))

    # L1: input bound from eps, using BN-folded row norm of conv1
    V1=rc["V1"][0]; s1c=rc["s1"][0]; s1p=rp["s1"][0]
    Co,H,W,_=V1.shape
    rn1=model.row_norm_conv(model.c1, model.bn1)                       # [32]
    B1=(eps*rn1.double().view(Co,1,1,1)*tfac.view(1,1,1,T)).expand(Co,H,W,T)
    tau1=lfp((V1.double()-theta).abs(), B1, beta, theta); e1=(s1p-s1c).abs().double()
    res["L1"]=(int(((tau1==0)&(e1==1)).sum()), float((1-tau1).mean()))

    # L2: propagate tau1 through BN-folded |conv2| (stride 2)
    V2=rc["V2"][0]; s2c=rc["s2"][0]; s2p=rp["s2"][0]
    w2=model.eff_conv(model.c2, model.bn2).abs().double()
    B2=prop_conv(tau1, w2, 2, 1, beta)
    tau2=lfp((V2.double()-theta).abs(), B2, beta, theta); e2=(s2p-s2c).abs().double()
    res["L2"]=(int(((tau2==0)&(e2==1)).sum()), float((1-tau2).mean()))

    # L3: propagate tau2 through BN-folded |conv3| (stride 2)
    V3=rc["V3"][0]; s3c=rc["s3"][0]; s3p=rp["s3"][0]
    w3=model.eff_conv(model.c3, model.bn3).abs().double()
    B3=prop_conv(tau2, w3, 2, 1, beta)
    tau3=lfp((V3.double()-theta).abs(), B3, beta, theta); e3=(s3p-s3c).abs().double()
    res["L3"]=(int(((tau3==0)&(e3==1)).sum()), float((1-tau3).mean()))

    # L4 (fc1): flatten tau3, propagate through |fc1|
    Vf=rc["Vf"][0]; sfc=rc["sf"][0]; sfp=rp["sf"][0]
    Bf=prop_fc(tau3.reshape(-1,T), model.fc1.weight.abs().double(), beta)
    tauf=lfp((Vf.double()-theta).abs(), Bf, beta, theta); ef=(sfp-sfc).abs().double()
    res["L4"]=(int(((tauf==0)&(ef==1)).sum()), float((1-tauf).mean()))

    # ---- Prop 3 on the NON-spiking readout ----
    # last hidden spike COUNTS (clean) and certified count-change bound (Prop 2)
    cnt_c = sfc.sum(-1).double()                       # [256] clean counts
    dcount = tauf.sum(-1).double()                     # [256] max count change per neuron
    Wro = model.readout.weight.double()                # [n_classes,256]
    bro = model.readout.bias.double()
    # readout uses rate = counts/T ; logit_c = sum_i Wro[c,i]*cnt_i/T + b_c
    logit = (Wro @ cnt_c)/T + bro
    top = int(logit.argmax())
    # worst-case logit erosion: each class logit can move by sum_i |Wro[c,i]|*dcount_i/T
    swing = (Wro.abs() @ dcount)/T                     # [n_classes] max |Δlogit_c|
    # top's logit can DROP by swing[top]; any other can RISE by swing[c].
    M = logit.clone()
    cert = True
    for c in range(logit.shape[0]):
        if c==top: continue
        # preserved if (logit[top]-swing[top]) > (logit[c]+swing[c])
        if (logit[top]-swing[top]) <= (logit[c]+swing[c]):
            cert=False; break
    # actual robustness referee
    rp_cnt = sfp.sum(-1).double()
    logit_p = (Wro @ rp_cnt)/T + bro
    act = bool(int(logit_p.argmax())==top)
    out_viol = int(cert and not act)
    res["out"]=(bool(cert), act, out_viol)
    return res


def build_model(ds, ckpt, dev):
    inch = 3 if ds=="cifar10" else 1
    img = 32 if ds=="cifar10" else 28
    m = ConvSNN(inch, img=img).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev)); m.eval()
    return m, inch, img

def get_testset(ds):
    from torchvision import datasets, transforms
    if ds=="cifar10":
        tfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,)*3,(0.5,)*3)]); DS=datasets.CIFAR10
    else:
        tfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
        DS={"mnist":datasets.MNIST,"fashion":datasets.FashionMNIST}[ds]
    return DS("./data", train=False, download=True, transform=tfm)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-glob", default="ckpt_*_T*.pt")
    args=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"

    results=[]
    for ckpt in sorted(glob.glob(args.ckpt_glob)):
        base=os.path.basename(ckpt).replace("ckpt_","").replace(".pt","")
        ds, Tt = base.rsplit("_T",1); T=int(Tt)
        model,inch,img=build_model(ds, ckpt, dev)
        ds_te=get_testset(ds)
        xs=torch.stack([ds_te[i][0] for i in range(args.n_samples)]).to(dev)
        viol={f"L{l}":0 for l in range(1,5)}; safe={f"L{l}":[] for l in range(1,5)}
        cert=act=outv=0
        for i in range(xs.size(0)):
            x=xs[i:i+1]; xp=x+l2_gauss(x,args.eps,seed=args.seed+i)
            r=certify(model,x,xp,T,args.eps,args.theta,args.beta)
            for l in range(1,5): viol[f"L{l}"]+=r[f"L{l}"][0]; safe[f"L{l}"].append(r[f"L{l}"][1])
            cert+=int(r["out"][0]); act+=int(r["out"][1]); outv+=r["out"][2]
        n=xs.size(0)
        rec={"ds":ds,"T":T,"eps":args.eps,
             "viol_per_layer":viol,"out_viol":outv,
             "total_viol":int(sum(viol.values())+outv),
             "certified_safe_by_layer_pct":{l:round(100*float(np.mean(safe[l])),2) for l in safe},
             "output_certified_robust_pct":round(100*cert/n,2),
             "output_actual_robust_pct":round(100*act/n,2)}
        results.append(rec)
        print(f"{ds:8s} T={T} | total_viol={rec['total_viol']:<2} | "
              f"safe L1-4=[{rec['certified_safe_by_layer_pct']['L1']:.0f},"
              f"{rec['certified_safe_by_layer_pct']['L2']:.0f},"
              f"{rec['certified_safe_by_layer_pct']['L3']:.0f},"
              f"{rec['certified_safe_by_layer_pct']['L4']:.0f}]% | "
              f"out cert-rob={rec['output_certified_robust_pct']:.0f}% "
              f"act-rob={rec['output_actual_robust_pct']:.0f}%", flush=True)

    print("\n==== COPY BELOW ===="); print(json.dumps(results,indent=2)); print("==== COPY ABOVE ====")
    with open("certify_results.json","w") as f: json.dump(results,f,indent=2)
    tot=sum(r["total_viol"] for r in results)
    print(f"\nTOTAL VIOLATIONS (must be 0): {tot}  ({'PASS' if tot==0 else 'FAIL'})")


if __name__=="__main__":
    main()
