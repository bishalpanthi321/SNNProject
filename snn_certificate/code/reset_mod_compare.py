#!/usr/bin/env python3
"""
reset_mod_compare.py
Reset-by-subtraction vs reset-to-mod certificate comparison (multilayer conv-SNN).
Prints a JSON summary at the end (copy-paste back for analysis). Saves reset_mod_results.json.

HONEST NOTE: the mod certificate reuses the SOUND binary reset term. Its validity is
gated by viol==0 and Rmax. If Rmax>1 with viol>0, those mod numbers are unsound and must
be discarded (binary term cannot bound multi-threshold crossings). If Rmax==1 everywhere,
mod == subtraction exactly.
"""

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
print("device:", dev)

# ============ CONFIG ============
DATASETS  = ["mnist", "fashion", "cifar10"]
TIMESTEPS = [4, 8]
BETAS     = [0.9]          # add 0.95, 0.99 if you want (multiplies runtime)
RESETS    = ["sub", "mod"]
EPOCHS    = 8
BATCH     = 128
THETA     = 1.0
EPS       = 0.1
N_SAMPLES = 30
ATTACK    = "gaussian"     # "gaussian" | "fgsm"
FAKE      = False          # True = quick smoke test on random data

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
    def __init__(self, in_ch, reset="sub", beta=0.9, theta=1.0, img=28, n_classes=10):
        super().__init__()
        self.beta, self.theta, self.reset = beta, theta, reset
        self.c1 = nn.Conv2d(in_ch, 16, 3, 2, 1, bias=False)
        self.c2 = nn.Conv2d(16, 32, 3, 2, 1, bias=False)
        s = img // 4
        self.fc1 = nn.Linear(32*s*s, 128, bias=False)
        self.fc_out = nn.Linear(128, n_classes, bias=False)
    def _rcount(self, V):
        if self.reset == "sub":
            return (V >= self.theta).float()
        return torch.clamp(torch.floor(V/self.theta), min=0.0)
    def _step(self, drive, V, rprev):
        V = self.beta*V + drive - self.theta*rprev
        s = spike(V, self.theta)
        return V, s, self._rcount(V)
    def forward(self, x, T, record=False):
        V1=V2=Vf=Vo=None; r1=r2=rf=ro=None; out=0.0
        rec={k:[] for k in ["V1","s1","r1","V2","s2","r2","Vf","sf","rf","Vo","so","ro"]} if record else None
        for _ in range(T):
            d=self.c1(x)
            if V1 is None: V1=torch.zeros_like(d); r1=torch.zeros_like(d)
            V1,s1,r1=self._step(d,V1,r1)
            d=self.c2(s1)
            if V2 is None: V2=torch.zeros_like(d); r2=torch.zeros_like(d)
            V2,s2,r2=self._step(d,V2,r2)
            f=s2.flatten(1); d=self.fc1(f)
            if Vf is None: Vf=torch.zeros_like(d); rf=torch.zeros_like(d)
            Vf,sf,rf=self._step(d,Vf,rf)
            d=self.fc_out(sf)
            if Vo is None: Vo=torch.zeros_like(d); ro=torch.zeros_like(d)
            Vo,so,ro=self._step(d,Vo,ro)
            out=out+so
            if record:
                for k,v in [("V1",V1),("s1",s1),("r1",r1),("V2",V2),("s2",s2),("r2",r2),
                            ("Vf",Vf),("sf",sf),("rf",rf),("Vo",Vo),("so",so),("ro",ro)]:
                    rec[k].append(v)
        if record:
            for k in rec: rec[k]=torch.stack(rec[k],-1)
            return out, rec
        return out

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
def l2_fgsm(model,x,y,eps,T):
    x=x.clone().detach().requires_grad_(True)
    loss=F.cross_entropy(model(x,T),y); g=torch.autograd.grad(loss,x)[0]
    fl=g.flatten(1); return (x+(fl/fl.norm(dim=1,keepdim=True).clamp_min(1e-12)*eps).view_as(x)).detach()
def prop_conv(tau_up, w_abs, stride, pad, beta):
    xx=tau_up.permute(3,0,1,2)
    A=F.conv2d(xx,w_abs,stride=stride,padding=pad).permute(1,2,3,0)
    return disc_input(A,beta)
def prop_fc(tau_flat, w_abs, beta):
    return disc_input(w_abs @ tau_flat, beta)

def get_data(ds, batch, fake):
    inch = 3 if ds=="cifar10" else 1; img = 32 if ds=="cifar10" else 28
    if fake:
        tr=torch.utils.data.TensorDataset(torch.rand(256,inch,img,img),torch.randint(0,10,(256,)))
        te=torch.utils.data.TensorDataset(torch.rand(64,inch,img,img),torch.randint(0,10,(64,)))
        return (torch.utils.data.DataLoader(tr,batch,shuffle=True),
                torch.utils.data.DataLoader(te,batch,shuffle=False),inch,img)
    from torchvision import datasets, transforms
    if ds=="cifar10":
        tfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,)*3,(0.5,)*3)]); DS=datasets.CIFAR10
    else:
        tfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
        DS={"mnist":datasets.MNIST,"fashion":datasets.FashionMNIST}[ds]
    tr=DS("./data",train=True,download=True,transform=tfm); te=DS("./data",train=False,download=True,transform=tfm)
    return (torch.utils.data.DataLoader(tr,batch,shuffle=True),
            torch.utils.data.DataLoader(te,batch,shuffle=False),inch,img)
def train(model,loader,T,epochs):
    opt=torch.optim.Adam(model.parameters(),1e-3); model.train()
    for ep in range(epochs):
        for xb,yb in loader:
            xb,yb=xb.to(dev),yb.to(dev)
            loss=F.cross_entropy(model(xb,T),yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
@torch.no_grad()
def accuracy(model,loader,T,maxb=20):
    c=n=0
    for i,(xb,yb) in enumerate(loader):
        xb,yb=xb.to(dev),yb.to(dev)
        c+=(model(xb,T).argmax(1)==yb).sum().item(); n+=yb.size(0)
        if i+1>=maxb: break
    return c/max(n,1)

@torch.no_grad()
def certify(model, x, xp, T, eps, theta, beta):
    _,rc=model(x,T,record=True); _,rp=model(xp,T,record=True)
    res={}; tfac=((1-beta**torch.arange(1,T+1,device=x.device).double())/(1-beta)); Rmax=0.0
    V1=rc["V1"][0]; s1c=rc["s1"][0]; s1p=rp["s1"][0]
    Co,H,W,_=V1.shape; rn1=model.c1.weight.flatten(1).norm(dim=1)
    B1=(eps*rn1.double().view(Co,1,1,1)*tfac.view(1,1,1,T)).expand(Co,H,W,T)
    tau1=lfp((V1.double()-theta).abs(),B1,beta,theta); e1=(s1p-s1c).abs().double()
    res["L1"]=(int(((tau1==0)&(e1==1)).sum()),float((1-tau1).mean()))
    V2=rc["V2"][0]; s2c=rc["s2"][0]; s2p=rp["s2"][0]
    B2=prop_conv(tau1,model.c2.weight.abs().double(),2,1,beta)
    tau2=lfp((V2.double()-theta).abs(),B2,beta,theta); e2=(s2p-s2c).abs().double()
    res["L2"]=(int(((tau2==0)&(e2==1)).sum()),float((1-tau2).mean()))
    Vf=rc["Vf"][0]; sfc=rc["sf"][0]; sfp=rp["sf"][0]
    Bf=prop_fc(tau2.reshape(-1,T),model.fc1.weight.abs().double(),beta)
    tauf=lfp((Vf.double()-theta).abs(),Bf,beta,theta); ef=(sfp-sfc).abs().double()
    res["L3"]=(int(((tauf==0)&(ef==1)).sum()),float((1-tauf).mean()))
    Vo=rc["Vo"][0]; soc=rc["so"][0]; sop=rp["so"][0]
    Bo=prop_fc(tauf,model.fc_out.weight.abs().double(),beta)
    tauo=lfp((Vo.double()-theta).abs(),Bo,beta,theta); eo=(sop-soc).abs().double()
    res["L4"]=(int(((tauo==0)&(eo==1)).sum()),float((1-tauo).mean()))
    if model.reset=="mod":
        for k in ["r1","r2","rf","ro"]:
            Rmax=max(Rmax,float(rc[k][0].max().item()),float(rp[k][0].max().item()))
    Cc=soc.sum(-1).double(); top=int(Cc.argmax())
    oth=Cc.clone(); oth[top]=-1e9; M=float(Cc[top]-Cc[oth.argmax()])
    dplus=float(tauo[top].sum()); dminus=float(max(tauo[c].sum() for c in range(Cc.shape[0]) if c!=top))
    cert=bool(M>dplus+dminus); Cp=sop.sum(-1).double(); act=bool(int(Cp.argmax())==top)
    res["out"]=(cert,act,int(cert and not act))
    casc=[float(e1.mean()),float(e2.mean()),float(ef.mean()),float(eo.mean())]
    return res, Rmax, casc

def run(ds,T,beta,reset):
    tr,te,inch,img=get_data(ds,BATCH,FAKE)
    model=ConvSNN(inch,reset=reset,beta=beta,theta=THETA,img=img).to(dev)
    train(model,tr,T,EPOCHS); acc=accuracy(model,te,T)
    xb,yb=next(iter(te)); xb,yb=xb.to(dev)[:N_SAMPLES],yb.to(dev)[:N_SAMPLES]
    viol={f"L{l}":0 for l in range(1,5)}; safe={f"L{l}":[] for l in range(1,5)}
    cert=act=outv=0; casc=np.zeros(4); Rmax=0.0
    for i in range(xb.size(0)):
        x=xb[i:i+1]
        xp = x+l2_gauss(x,EPS,seed=i) if ATTACK=="gaussian" else l2_fgsm(model,x,yb[i:i+1],EPS,T)
        r,rm,c=certify(model,x,xp,T,EPS,THETA,beta)
        Rmax=max(Rmax,rm); casc+=np.array(c)
        for l in range(1,5):
            viol[f"L{l}"]+=r[f"L{l}"][0]; safe[f"L{l}"].append(r[f"L{l}"][1])
        cert+=int(r["out"][0]); act+=int(r["out"][1]); outv+=r["out"][2]
    n=xb.size(0)
    return {"ds":ds,"T":T,"beta":beta,"reset":reset,"acc":acc,"Rmax":Rmax,
            "viol":{l:viol[l] for l in viol},
            "safe":{l:float(np.mean(safe[l])) for l in safe},
            "cert_frac":cert/n,"act_frac":act/n,"out_viol":outv,
            "cascade":(casc/n).tolist()}
results=[]
for ds in DATASETS:
    for T in TIMESTEPS:
        for beta in BETAS:
            for reset in RESETS:
                r=run(ds,T,beta,reset); results.append(r)
                print(f"{ds:8s} T={T} b={beta} {reset:3s} | acc={r['acc']:.3f} "
                      f"Rmax={r['Rmax']:.0f} viol={sum(r['viol'].values())+r['out_viol']:<2} "
                      f"| safe=[{100*r['safe']['L1']:.0f},{100*r['safe']['L2']:.0f},"
                      f"{100*r['safe']['L3']:.0f},{100*r['safe']['L4']:.0f}]% "
                      f"cert-rob={100*r['cert_frac']:.0f}%")
print("\nDONE")

print("SOUNDNESS: total viol (must be 0 to trust certified numbers)")
tot=sum(sum(r["viol"].values())+r["out_viol"] for r in results)
print(f"  total = {tot}  ({'PASS' if tot==0 else 'FAIL — some mod configs had multi-crossing; discard those'})\n")
print(f"{'config':22}{'reset':6}{'acc':>6}{'Rmax':>5}{'L1':>6}{'L2':>6}{'L3':>6}{'L4':>6}{'cert':>7}")
for ds in DATASETS:
    for T in TIMESTEPS:
        for beta in BETAS:
            for reset in RESETS:
                r=next(x for x in results if x["ds"]==ds and x["T"]==T and x["beta"]==beta and x["reset"]==reset)
                print(f"{ds+' T'+str(T)+' b'+str(beta):22}{reset:6}{r['acc']:>6.2f}{r['Rmax']:>5.0f}"
                      f"{100*r['safe']['L1']:>5.0f}%{100*r['safe']['L2']:>5.0f}%"
                      f"{100*r['safe']['L3']:>5.0f}%{100*r['safe']['L4']:>5.0f}%{100*r['cert_frac']:>6.0f}%")

# ---- DUMP EVERYTHING AS JSON (copy this whole output back) ----
import json
summary = {
    "config": {"datasets": DATASETS, "timesteps": TIMESTEPS, "betas": BETAS,
               "resets": RESETS, "epochs": EPOCHS, "eps": EPS, "theta": THETA,
               "n_samples": N_SAMPLES, "attack": ATTACK},
    "total_viol": int(sum(sum(r["viol"].values()) + r["out_viol"] for r in results)),
    "runs": []
}
for r in results:
    summary["runs"].append({
        "ds": r["ds"], "T": r["T"], "beta": r["beta"], "reset": r["reset"],
        "acc": round(r["acc"], 4),
        "Rmax": r["Rmax"],
        "viol_per_layer": r["viol"],
        "out_viol": r["out_viol"],
        "total_viol_this_run": int(sum(r["viol"].values()) + r["out_viol"]),
        "certified_safe_by_layer_pct": {l: round(100*r["safe"][l], 2) for l in r["safe"]},
        "output_certified_robust_pct": round(100*r["cert_frac"], 2),
        "output_actual_robust_pct": round(100*r["act_frac"], 2),
        "empirical_flip_rate_by_layer_pct": [round(100*c, 3) for c in r["cascade"]],
    })

print("==== COPY EVERYTHING BELOW THIS LINE ====")
print(json.dumps(summary, indent=2))
print("==== COPY EVERYTHING ABOVE THIS LINE ====")

# also save to file in case you want it
with open("reset_mod_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n(also saved to reset_mod_results.json)")
