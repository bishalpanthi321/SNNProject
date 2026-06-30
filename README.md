# A Deterministic Least Fixed-Point Certificate for Spike Preservation in SNNs

Deterministic, per-neuron, per-timestep certificate of **spike preservation** for trained
leaky integrate-and-fire (LIF) spiking neural networks under **reset-by-subtraction**
dynamics and `L2`-bounded input perturbation.

The certificate is the **least fixed point** of a monotone operator on the Boolean lattice
`{0,1}^T`. Existence follows from the Knaster–Tarski theorem; soundness from a causal
temporal induction (Theorem 1 in the paper). It is **attack-agnostic**: the certified-safe
set is identical for every `L2`-bounded perturbation of a given radius.

---

## What is validated (all with zero soundness violations)

| Result | Where | Status |
|---|---|---|
| Single-layer certificate, 27 configs at eps=0.1 | `results/` (grid in paper) | **viol=0**, 85–93% certified-safe |
| eps-sweep (0.1→4.0), Gaussian | `results/epssweep_gaussian.json` | **viol=0** all 6 points/dataset |
| eps-sweep, FGSM adversarial | `results/epssweep_fgsm.json` | **viol=0** all 18 points; attack-agnostic |
| Compositional soundness across 4 layers (trained nets) | `results/multilayer_certify_results.json` | **viol=0** every layer, n=300 |

A single figure summarizing all of the above: `figures/results_summary.png`.

## What is NOT validated (kept for honesty, do not cite)

See `results/EXPLORATORY_not_validated.md`. Briefly:
- **Back-substitution / symbolic propagation** (`code/certify_backsub.py`): leaked soundness
  (viol=2). Promising but needs a sound CROWN-style relaxation of the reset-coupled flags.
- **Reset-to-mod** (`code/reset_mod_compare.py`): inconclusive / likely does not help the
  pointwise certificate; only one trained multi-crossing example observed.

---

## Directory layout

```
snn_certificate/
├── README.md
├── code/
│   # --- single-layer certificate (the core, validated result) ---
│   ├── lfp_certificate.py             # original per-neuron certificate (base reference)  [VALIDATED]
│   ├── deep_core.py                   # shared engine for the per-dataset sweeps          [VALIDATED]
│   ├── run_mnist.py                   # thin wrapper: MNIST sweep (imports deep_core)     [VALIDATED]
│   ├── run_fashion_mnist.py           # thin wrapper: Fashion-MNIST sweep                 [VALIDATED]
│   ├── run_cifar10.py                 # thin wrapper: CIFAR-10 sweep                       [VALIDATED]
│   ├── lfp_certificate_trained.py     # cert on TRAINED weights -> paper's 27-config grid [VALIDATED]
│   ├── lfp_certificate_epssweep.py    # cert + eps-sweep (Gaussian/FGSM)                  [VALIDATED]
│   # --- multilayer ---
│   ├── multilayer_train.py            # trains the 4-layer conv-SNN (BN + rate readout)   [VALIDATED]
│   ├── certify_checkpoints.py         # multilayer cert on trained ckpts (BN folded)      [VALIDATED]
│   ├── lfp_multilayer.py              # end-to-end multilayer chain (train+certify)
│   # --- exploratory (NOT sound / NOT for publication) ---
│   ├── reset_mod_compare.py           # reset-to-mod comparison                           [EXPLORATORY]
│   └── certify_backsub.py             # symbolic back-substitution                        [EXPLORATORY/UNSOUND]
├── results/
│   ├── epssweep_gaussian.json
│   ├── epssweep_fgsm.json
│   ├── training_results.json
│   ├── multilayer_certify_results.json
│   └── EXPLORATORY_not_validated.md
├── figures/
│   ├── fig_cascade.png         # the reset-cascade obstacle (paper Fig 1)
│   ├── fig_grid.png            # certified-safe over (beta,T)
│   ├── fig_degradation.png     # certified-safe vs eps
│   ├── fig_flip.png            # Gaussian vs FGSM flip-rate
│   ├── fig_depth.png           # compositional soundness across depth
│   └── results_summary.png     # one-page summary of all validated results
└── paper/
    ├── LFP_SNN_paper.pdf        # the manuscript
    ├── paper.tex
    └── refs.bib
```

---

## Reproducing the validated results

Environment: Python 3, `torch`, `torchvision`, `numpy`. A GPU is recommended.

### 1. Single-layer certificate + eps-sweep (the core result)
```bash
cd code
# Gaussian:
python lfp_certificate_epssweep.py --datasets mnist fashion cifar10 \
    --betas 0.9 --timesteps 20 --eps-sweep 0.1 0.25 0.5 1.0 2.0 4.0 \
    --attack gaussian --epochs 3 --n-samples 50
# FGSM (adversarial):
python lfp_certificate_epssweep.py --datasets mnist fashion cifar10 \
    --betas 0.9 --timesteps 20 --eps-sweep 0.1 0.25 0.5 1.0 2.0 4.0 \
    --attack fgsm --epochs 3 --n-samples 50
```
Expect: certified-safe % matching `results/epssweep_*.json`, and **Total violations: 0**.

### 2. Train the 4-layer conv-SNN
```bash
cd code
# fast sanity gate first (~2 min):
python multilayer_train.py --sanity
# full training (saves ckpt_<ds>_T<n>.pt):
python multilayer_train.py --datasets mnist fashion cifar10 --timesteps 4 8 --epochs 8
```
Expect: accuracies matching `results/training_results.json` (MNIST ~98–99%, Fashion ~90–92%,
CIFAR-10 ~70%).

### 3. Certify the trained checkpoints across depth
```bash
cd code
python certify_checkpoints.py --eps 0.1 --n-samples 300
```
Expect: per-layer certified-safe % matching `results/multilayer_certify_results.json`, and
**TOTAL VIOLATIONS (must be 0): 0 (PASS)** — soundness composes across all four layers.

---

## The core math (one paragraph)

For a reset-by-subtraction LIF layer, a single perturbation-induced spike flip injects a
discrepancy that the reset couples forward in time, so the bound at time `t` depends on the
(unknown) earlier flips. We replace the unknown flips with a binary risk-flag vector `tau`
and define a monotone operator `F` on `{0,1}^T` whose argument adds the reset term
`theta * sum_k beta^(t-k) tau(k-1)` to the clean displacement bound
`B_input = eps * ||w_i|| * (1 - beta^t)/(1 - beta)`. The least fixed point of `F` (computed by
upward Kleene iteration, `<= T` steps) is the tightest sound certificate: any neuron it flags
safe provably cannot flip under any `L2`-bounded perturbation of the given radius.

---

## Honesty notes

- `viol` (a certified-safe spike that actually flipped) is the ground-truth soundness gate in
  every script. **Any result with viol > 0 is not a certificate** and is excluded from the
  validated set above.
- The certified *fraction* reduces with depth (Table in the paper). The certificate stays
  **sound** at depth, but is not yet non-vacuous at the output for deep nets; tightening the
  propagation is the stated open problem.
- Bibliographic details in `paper/refs.bib` should be verified against original sources before
  submission.
