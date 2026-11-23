"""Correlation vs. Causation: does discounting cause profit loss in Superstore?

Design
------
1. Where does the data live?   Discount support (what range can we say anything about?)
2. Is there confounding?       Balance of PRE-discount characteristics between
                               high- and low-discount lines (standardized differences).
3. Estimate the effect.        OLS ladder with heteroskedasticity-robust (HC3) SEs, adding
                               only pre-treatment controls; plus one deliberately "bad"
                               model that controls for post-treatment variables (Sales,
                               Order Quantity) to show how that biases the answer.
4. Robustness.                 Median (quantile) regression - profit per line is heavy-tailed.
5. Does discount buy volume?   Discount vs Order Quantity / Sales.

Run:  python analysis.py      (input: superstore_clean.csv from superstore-sales-analysis)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path(__file__).resolve().parent
DARK, ACCENT, GREY = "#264653", "#e76f51", "#8d99ae"
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlelocation": "left", "axes.titleweight": "bold", "font.size": 10,
                     "text.parse_math": False})

df = pd.read_csv(WORK / "superstore_clean.csv", parse_dates=["Order Date"])
df.columns = [c.replace(" ", "_").replace("%", "pct") for c in df.columns]
df["Year"] = df.Order_Date.dt.year
n = len(df)
PER = 10  # report effects per 10 percentage points of discount

# ---- 1. Support of the data -------------------------------------------------
support = {
    "lines": n,
    "lines_0_to_10pct": int((df.Discount <= 0.10).sum()),
    "lines_above_10pct": int((df.Discount > 0.10).sum()),
    "lines_above_20pct": int((df.Discount > 0.20).sum()),
    "max_discount": float(df.Discount.max()),
    "mean_discount": round(float(df.Discount.mean()), 4),
}

# ---- 2. Pooled correlations -------------------------------------------------
num_cols = ["Discount", "Order_Quantity", "Product_Base_Margin", "Unit_Price", "Profit", "Sales", "Shipping_Cost"]
corr = df[num_cols].corr()
corr.round(3).to_csv(WORK / "correlation_matrix.csv")
r = corr.loc["Discount", "Profit"]
t = r * np.sqrt((n - 2) / (1 - r ** 2))
from scipy import stats  # noqa: E402  (statsmodels depends on scipy)
p_r = 2 * stats.t.sf(abs(t), n - 2)

# ---- 3. Balance: are high-discount lines different BEFORE the discount? -----
hi = df.Discount > df.Discount.median()
bal_rows = []
for col in ["Product_Base_Margin", "Unit_Price", "Shipping_Cost"]:
    a, b = df.loc[hi, col], df.loc[~hi, col]
    bal_rows.append((col.replace("_", " "), (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)))
for col in ["Department", "Category", "Region", "Customer_Segment", "Year"]:
    for lvl in df[col].unique():
        x = (df[col] == lvl).astype(float)
        a, b = x[hi], x[~hi]
        sd = np.sqrt((a.var() + b.var()) / 2)
        if sd > 0:
            bal_rows.append((f"{col.replace('_', ' ')}: {lvl}", (a.mean() - b.mean()) / sd))
bal = pd.DataFrame(bal_rows, columns=["covariate", "std_mean_diff"])
bal["abs"] = bal.std_mean_diff.abs()
bal.sort_values("abs", ascending=False).drop(columns="abs").round(4).to_csv(WORK / "balance_table.csv", index=False)
dept = df.groupby("Department").agg(lines=("Profit", "size"), mean_discount=("Discount", "mean"),
                                    sales=("Sales", "sum"), profit=("Profit", "sum"))
dept["margin_pct"] = dept.profit / dept.sales * 100
dept.round(4).to_csv(WORK / "discount_by_department.csv")

# ---- 4. Regression ladder (HC3 robust SEs) ----------------------------------
PRE = "C(Category) + Product_Base_Margin + Unit_Price"
CTX = "C(Region) + C(Customer_Segment) + C(Year)"
specs = [
    ("1. Naive: Discount only", "Profit ~ Discount", "ok"),
    ("2. + department", "Profit ~ Discount + C(Department)", "ok"),
    ("3. + sub-category (17)", "Profit ~ Discount + C(Category)", "ok"),
    ("4. + base margin, unit price", f"Profit ~ Discount + {PRE}", "ok"),
    ("5. + region, segment, year", f"Profit ~ Discount + {PRE} + {CTX}", "main"),
    ("6. BAD: + Sales, Order Qty", f"Profit ~ Discount + {PRE} + {CTX} + Sales + Order_Quantity", "bad"),
]
rows = []
for name, f, kind in specs:
    m = smf.ols(f, df).fit(cov_type="HC3")
    lo, hi_ = m.conf_int().loc["Discount"]
    rows.append({"model": name, "kind": kind, "effect_per_10pp": m.params["Discount"] / PER,
                 "ci_low": lo / PER, "ci_high": hi_ / PER, "p_value": m.pvalues["Discount"],
                 "r_squared": m.rsquared})
ladder = pd.DataFrame(rows)
ladder.round(4).to_csv(WORK / "naive_vs_controlled.csv", index=False)
naive, main, bad = ladder.iloc[0], ladder[ladder.kind == "main"].iloc[0], ladder[ladder.kind == "bad"].iloc[0]

# ---- 5. Robustness: median regression ----------------------------------------
# (17 sub-category dummies stop the median solver converging; department-level controls do converge)
q = smf.quantreg("Profit ~ Discount + C(Department) + Product_Base_Margin + Unit_Price", df).fit(q=0.5, max_iter=20000)
q_lo, q_hi = q.conf_int().loc["Discount"]

# ---- 6. Does discount buy volume? -------------------------------------------
vol = smf.ols(f"Order_Quantity ~ Discount + {PRE} + {CTX}", df).fit(cov_type="HC3")

# ---- 7. Bonus: pooled vs within-group correlation sign flip ------------------
tech = df[df.Department == "Technology"]
flip = {"pooled_shipcost_vs_base_margin": round(float(df.Shipping_Cost.corr(df.Product_Base_Margin)), 3),
        "technology_only": round(float(tech.Shipping_Cost.corr(tech.Product_Base_Margin)), 3)}

# Implied cost of current discounting IF the main estimate were causal (linear, within 0-10%)
implied = float(main.effect_per_10pp * PER * df.Discount.sum())
summary = {
    "support": support,
    "pearson_discount_profit": round(float(r), 4), "pearson_p": round(float(p_r), 6),
    "effect_per_10pp_naive": round(float(naive.effect_per_10pp), 1),
    "effect_per_10pp_adjusted": round(float(main.effect_per_10pp), 1),
    "effect_per_10pp_adjusted_ci95": [round(float(main.ci_low), 1), round(float(main.ci_high), 1)],
    "effect_per_10pp_bad_control_model": round(float(bad.effect_per_10pp), 1),
    "change_naive_to_adjusted_pct": round(float((main.effect_per_10pp / naive.effect_per_10pp - 1) * 100), 1),
    "max_abs_std_mean_diff": round(float(bal["abs"].max()), 3),
    "mean_discount_by_department": dept.mean_discount.round(4).to_dict(),
    "margin_pct_by_department": dept.margin_pct.round(1).to_dict(),
    "median_regression_effect_per_10pp": round(float(q.params["Discount"] / PER), 1),
    "median_regression_ci95": [round(float(q_lo / PER), 1), round(float(q_hi / PER), 1)],
    "order_qty_effect_per_10pp": round(float(vol.params["Discount"] / PER), 2),
    "order_qty_effect_p": round(float(vol.pvalues["Discount"]), 3),
    "implied_profit_cost_of_all_discounts_if_causal": round(implied, 0),
    "implied_cost_ci95": [round(float(main.ci_high * PER * df.Discount.sum()), 0),
                          round(float(main.ci_low * PER * df.Discount.sum()), 0)],
    "sign_flip_example": flip,
}
(WORK / "causation_summary.json").write_text(json.dumps(summary, indent=2))

# ---- Charts -------------------------------------------------------------------
# A. Coefficient ladder (forest plot)
fig, ax = plt.subplots(figsize=(9, 4.3))
lad = ladder.iloc[::-1].reset_index(drop=True)
for i, rw in lad.iterrows():
    c = ACCENT if rw.kind == "bad" else (DARK if rw.kind == "main" else GREY)
    ax.errorbar(rw.effect_per_10pp, i, xerr=[[rw.effect_per_10pp - rw.ci_low], [rw.ci_high - rw.effect_per_10pp]],
                fmt="o", color=c, capsize=3, lw=1.5)
    ax.text(rw.ci_high + 4, i, f"{rw.effect_per_10pp:.0f}", va="center", fontsize=8, color=c)
ax.set_yticks(range(len(lad)), lad.model)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Change in profit per line ($) for +10 points of discount, with 95% CI")
ax.set_title(f"Adjusting for pre-discount factors leaves the effect unchanged (about {main.effect_per_10pp:.0f});\n"
             f"only controlling for Sales/Qty, which the discount itself affects, shrinks it")
fig.tight_layout()
fig.savefig(WORK / "naive_vs_controlled.png")
plt.close(fig)

# B. Balance plot
top = bal.sort_values("abs", ascending=False).head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh(top.covariate, top.std_mean_diff, color=DARK)
ax.axvspan(-0.1, 0.1, color=GREY, alpha=0.15, lw=0)
ax.text(0.098, -0.9, "|d| < 0.1 = balanced", ha="right", fontsize=8, color=GREY)
ax.set_xlim(-0.25, 0.25)
ax.set_xlabel("Standardized mean difference, lines with above- vs below-median discount")
ax.set_title(f"No confounding visible: high- and low-discount lines look alike\n"
             f"on every pre-discount factor (largest |d| = {bal['abs'].max():.2f}; 15 largest shown)")
fig.tight_layout()
fig.savefig(WORK / "balance_check.png")
plt.close(fig)

# C. Where the data lives
fig, ax = plt.subplots(figsize=(9, 3.6))
vc = df.Discount.round(2).value_counts().sort_index()
ax.bar(vc.index * 100, vc.values, width=0.8, color=[DARK if v <= 10 else ACCENT for v in vc.index * 100])
for v, c in vc[vc.index > 0.10].items():
    ax.text(v * 100, c + 25, str(c), ha="center", fontsize=8, color=ACCENT)
ax.set_xlabel("Discount (%)")
ax.set_ylabel("Order lines")
ax.set_title(f"{support['lines_0_to_10pct']:,} of {n:,} lines are discounted 0-10%: "
             f"no evidence at all about deeper discounts")
fig.tight_layout()
fig.savefig(WORK / "discount_distribution.png")
plt.close(fig)

print(ladder.round(2).to_string(index=False))
print(json.dumps(summary, indent=2))
