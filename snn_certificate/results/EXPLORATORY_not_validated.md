# Exploratory results — NOT validated, NOT for publication

These two directions were explored but are **not sound / not solid**, and are kept here
only for honesty and future work. **Do not cite these numbers as results.**

## 1. Back-substitution (symbolic propagation) — SOUNDNESS LEAK
- Idea: compose linear maps before taking the norm (CROWN-style) to fight depth looseness.
- Offline the core identity is sound and ~3x tighter; BUT the implemented certificate
  produced **symbolic_viol = 2** on fashion-T4 (total_viol = 2, FAIL).
- Root cause: the symbolic bound dropped the upstream reset-discrepancy contribution.
  A sound fix requires a CROWN-style *relaxation of the reset-coupled flags*, which is a
  real derivation (not done). The large apparent tightening (L4 0% -> ~30%) was partly a
  consequence of the unsound (too-small) bound.
- Status: PROMISING DIRECTION, NOT SOUND. See code/certify_backsub.py.

## 2. Reset-to-mod comparison — INCONCLUSIVE / likely negative
- Hypothesis: reset-to-mod cascades less than reset-by-subtraction.
- Result: only ONE trained model reached the multi-crossing regime (fashion T=8, Rmax=8),
  and there mod made the cascade WORSE (highest flip rate, fastest collapse). Most configs
  had Rmax=1 (mod == subtraction exactly). total_viol=0 only because Rmax stayed bounded.
- Consistent with the math: mod's advantage is about accumulated-signal error (Alexiewicz
  norm), NOT the pointwise spike-flip bound. A single trained multi-crossing example is not
  enough to make any claim.
- Status: DOES NOT HELP (and can hurt) the pointwise certificate; one data point only.
