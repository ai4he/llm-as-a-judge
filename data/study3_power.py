#!/usr/bin/env python3
"""Study 3 sample-size / statistical-power analysis.

Justifies per-dataset sample sizes for the LLM-as-a-judge evaluation so that
agreement estimates are precise and differences vs. the human ceiling are
detectable. Pure computation (no API). Outputs a table + writes
data/study3_power.json for the protocol.
"""
import json, math
from pathlib import Path
from scipy.stats import norm
HERE = Path(__file__).resolve().parent
za = norm.ppf(0.975)          # 1.95996 (two-sided 95% CI / alpha=0.05)
zb = norm.ppf(0.80)           # 0.84162 (power = 0.80)

def n_prop_ci(p, w):
    "n for a 95% CI half-width w on a proportion p (e.g., % agreement / accuracy)."
    return math.ceil(za**2 * p*(1-p) / w**2)

def n_two_prop(p1, p2, power=0.80, alpha=0.05):
    "n PER GROUP to detect p1 vs p2 (independent), two-sided. Conservative vs paired."
    za_=norm.ppf(1-alpha/2); zb_=norm.ppf(power); pbar=(p1+p2)/2
    return math.ceil((za_*math.sqrt(2*pbar*(1-pbar)) + zb_*math.sqrt(p1*(1-p1)+p2*(1-p2)))**2 / (p1-p2)**2)

def n_mcnemar(p10, p01, power=0.80, alpha=0.05):
    "Paired (same items): detect difference in agreement via discordant pairs (McNemar)."
    pdisc=p10+p01; psi=(p10-p01)
    za_=norm.ppf(1-alpha/2); zb_=norm.ppf(power)
    return math.ceil((za_*math.sqrt(pdisc) + zb_*math.sqrt(pdisc - psi**2))**2 / psi**2)

def n_corr(rho, w):
    "Approx n for 95% CI half-width w on Pearson/Spearman r via Fisher z."
    # CI on z is +- za/sqrt(n-3); invert at the point estimate (delta method approx)
    # width in r-space ~ (1-rho^2)*za/sqrt(n-3); solve for n
    return math.ceil((((1-rho**2)*za)/w)**2 + 3)

out = {"params": {"alpha":0.05,"power":0.80,"za":round(za,4),"zpower":round(zb,4)}, "tables": {}}

# (A) precision of an agreement proportion (worst case p=0.5; typical p=0.80)
ci = {}
for p in (0.50, 0.70, 0.80, 0.90):
    ci[f"p={p}"] = {f"halfwidth={w}": n_prop_ci(p, w) for w in (0.05, 0.04, 0.03, 0.02)}
out["tables"]["proportion_CI_n"] = ci

# (B) power to detect LLM-human vs human-human agreement gap (baseline human ceiling ~0.75)
gap = {}
for base in (0.70, 0.75, 0.80):
    row={}
    for d in (0.10, 0.075, 0.05):
        p1=base; p2=max(0.01,min(0.99,base-d))
        row[f"delta={d}"] = {"independent_per_group": n_two_prop(p1,p2),
                             "paired_mcnemar": n_mcnemar(p10=d*0.6, p01=d*0.6-d if d*0.6-d>0 else 0.02)}
    gap[f"human_ceiling={base}"]=row
out["tables"]["detect_gap_n"] = gap

# (C) correlation CI (ordinal/continuous tasks)
out["tables"]["correlation_CI_n"] = {f"rho={r}": {f"halfwidth={w}": n_corr(r,w) for w in (0.10,0.05)}
                                     for r in (0.5,0.7,0.8)}

# (D) Cohen's kappa precision (guideline): bootstrap CI half-width ~0.05 needs n on the order of
#     a few hundred for moderate kappa; the alt-test (leave-one-annotator-out) needs >=3 (ideally >=5)
#     human annotators per item. We state operational targets below.
out["recommendation"] = {
  "per_dataset_target_n": 500,
  "rationale": "n=500 gives <=+-4.4% 95% CI on an agreement proportion at p=0.5 (tighter at p>=0.8), "
               "and >=80% power to detect a 0.10 gap vs the human ceiling on the same items (paired). "
               "Use the FULL dataset when it has <=1000 labelable items; otherwise draw a stratified "
               "random sample of n=500 (by class label and, where available, by annotator-disagreement "
               "stratum) so rare classes and hard/contested items are represented.",
  "annotators_required": ">=3 human annotators per item for the human-human ceiling and the alt-test "
                         "(>=5 preferred for stable leave-one-out winning-rate estimates).",
  "replication": "temperature=0 for the primary run + 3 replicate runs at temperature=0.7 to quantify "
                 "self-consistency (intra-model agreement); >=3 prompt variants for prompt-sensitivity.",
  "multiplicity": "Benjamini-Hochberg FDR across the (dataset x model x metric) grid.",
  "total_scale": "With ~6 datasets x ~6 judge models x 500 items x (1 primary + 3 replicate) ~= 72k "
                 "primary + 216k replicate model judgments; ample for mixed-effects modelling.",
}

(HERE/"study3_power.json").write_text(json.dumps(out, indent=2))
print("=== (A) n for 95% CI half-width on an agreement proportion ===")
for p,row in ci.items():
    print(f"  {p}: " + "  ".join(f"+-{w.split('=')[1]}->{n}" for w,n in row.items()))
print("\n=== (B) n to detect a gap vs the human ceiling (power 0.80) ===")
for base,row in gap.items():
    print(f"  {base}:")
    for d,vals in row.items():
        print(f"     {d}: independent/group={vals['independent_per_group']}, paired(McNemar)={vals['paired_mcnemar']}")
print("\n=== (C) n for 95% CI half-width on a correlation ===")
for r,row in out["tables"]["correlation_CI_n"].items():
    print(f"  {r}: " + "  ".join(f"+-{w.split('=')[1]}->{n}" for w,n in row.items()))
print("\nRECOMMENDATION:", json.dumps(out["recommendation"], indent=2))
