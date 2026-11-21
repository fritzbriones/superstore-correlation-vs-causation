"""
Correlation vs. Causation: Does discounting cause profit loss in Superstore?

Run: python analysis.py
Outputs: correlation_matrix.csv, naive_vs_controlled.csv,
         simpson_by_subcategory.csv, causation_summary.json, charts/*.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path(__file__).resolve().parent
df = pd.read_csv(WORK / "superstore_clean.csv")

n = len(df)
# ---- 1. Naive pooled correlations -----------------------------------------
num_cols = ["Discount", "Order Quantity", "Product Base Margin", "Profit", "Sales", "Shipping Cost"]
corr = df[num_cols].corr(method="pearson")
corr.round(3).to_csv(WORK / "correlation_matrix.csv")

r_discount_profit = corr.loc["Discount", "Profit"]

# ---- 2. Confounding anatomy: who gets discounted? --------------------------
# Discount frequency and typical margin by sub-category ("Department" field)
dept = df.groupby("Department").agg(
    n=("Row ID", "count"),
    mean_discount=("Discount", "mean"),
    share_discounted=("Discount", lambda s: (s > 0).mean()),
    mean_margin=("Profit Margin %", "mean"),
    mean_profit=("Profit", "mean"),
).round(4)
dept.to_csv(WORK / "simpson_by_subcategory.csv")
# Pooled vs within-department discount/profit correlations
within_corr = df.groupby("Department").apply(
    lambda g: g["Discount"].corr(g["Profit"]) if len(g) > 30 else np.nan
)

# ---- 3. Proper test: naive vs controlled regression ------------------------
y = df["Profit"]
X_naive = sm.add_constant(df[["Discount"]])
m_naive = sm.OLS(y, X_naive).fit()

controls = pd.get_dummies(df["Department"], prefix="dept", drop_first=True).astype(float)
X_ctrl = pd.concat(
    [df[["Discount", "Sales", "Order Quantity", "Product Base Margin"]].astype(float), controls],
    axis=1,
)
m_ctrl = sm.OLS(y, sm.add_constant(X_ctrl)).fit()

comp = pd.DataFrame({
    "model": ["Naive (Discount only)", "Controlled (Dept FE + Sales + Qty + Margin)"],
    "discount_coef": [m_naive.params["Discount"], m_ctrl.params["Discount"]],
    "std_err": [m_naive.bse["Discount"], m_ctrl.bse["Discount"]],
    "p_value": [m_naive.pvalues["Discount"], m_ctrl.pvalues["Discount"]],
    "r_squared": [m_naive.rsquared, m_ctrl.rsquared],
})
comp.round(4).to_csv(WORK / "naive_vs_controlled.csv", index=False)

# ---- 4. Is y = f(x) actually monotone? does discount "cause" loss at all levels?
bins = pd.cut(df["Discount"], bins=[-0.01, 0.0, 0.10, 0.20, 0.30, 0.50, 1.0],
              labels=["0%", "1-10%", "11-20%", "21-30%", "31-50%", ">50%"])
by_disc = df.groupby(bins, observed=True).agg(
    n=("Profit", "count"),
    mean_profit=("Profit", "mean"),
    share_lines_unprofitable=("Profit", lambda s: (s < 0).mean()),
).round(4)
by_disc.to_csv(WORK / "profit_by_discount_bin.csv")

naive_beta = m_naive.params["Discount"]
ctrl_beta = m_ctrl.params["Discount"]
summary = {
    "rows": int(n),
    "pearson_discount_profit_pooled": round(float(r_discount_profit), 4),
    "naive_coef_discount": round(float(naive_beta), 4),
    "naive_p": round(float(m_naive.pvalues["Discount"]), 6),
    "controlled_coef_discount": round(float(ctrl_beta), 4),
    "controlled_p": round(float(m_ctrl.pvalues["Discount"]), 6),
    "share_of_naive_effect_explained_by_controls": round(1 - ctrl_beta / naive_beta, 4),
    "within_dept_corr_range": [round(float(within_corr.min()), 3), round(float(within_corr.max()), 3)],
    "within_dept_corr_median": round(float(within_corr.median()), 3),
    "most_discounted_depts": dept.sort_values("mean_discount", ascending=False).head(3).index.tolist(),
    "thinnest_margin_depts": dept.sort_values("mean_margin").head(3).index.tolist(),
}
(WORK / "causation_summary.json").write_text(json.dumps(summary, indent=2))

# ---- 5. Charts --------------------------------------------------------------
(WORK / "charts_previous_runs").mkdir(exist_ok=True)  # unused in flat layout

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(df["Discount"] + np.random.uniform(-0.005, 0.005, n), df["Profit"],
           s=6, alpha=0.25, color="#3944bc")
xs = np.linspace(0, df["Discount"].max(), 50)
ax.plot(xs, m_naive.params["const"] + naive_beta * xs, color="#c0392b", lw=2,
        label=f"Naive fit: slope = {naive_beta:.1f}")
ax.plot(xs, m_ctrl.params["const"] + ctrl_beta * xs, color="#1e8449", lw=2, ls="--",
        label=f"Controlled fit: slope = {ctrl_beta:.1f}")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Discount (0 = none, 0.2 = 20% off)")
ax.set_ylabel("Profit per line ($)")
ax.set_title("Same data, two stories: discount slope before vs. after controls")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(WORK / "naive_vs_controlled.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
top = dept.sort_values("mean_discount", ascending=False).head(8).index[::-1]
sub = dept.loc[top]
colors = plt.cm.viridis((sub["mean_discount"] / df["Discount"].max()).clip(0.15, 1))
ax.barh(range(len(sub)), sub["mean_margin"], color=colors)
ax.set_yticks(range(len(sub)), top)
ax.axvline(df["Profit Margin %"].mean(), color="#c0392b", ls="--", lw=1,
           label=f"Company avg margin = {df['Profit Margin %'].mean():.1f}%")
ax.set_xlabel("Mean profit margin (%)")
ax.set_title("Confounding anatomy: the most-discounted departments also run thin margins")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(WORK / "confounding_anatomy.png", dpi=150)
plt.close(fig)

print("=== Pooled correlations (Pearson) ===")
print(corr.round(3).to_string())
print("\n=== Discount anatomy by department (top by mean discount) ===")
print(dept.sort_values("mean_discount", ascending=False).head(6).to_string())
print("\n=== Naive vs controlled ===")
print(comp.round(4).to_string(index=False))
print("\n=== Profit by discount bin ===")
print(by_disc.to_string())
print("\nWithin-department Discount~Profit corr, median:", round(float(within_corr.median()), 3))
print(json.dumps(summary, indent=2))
