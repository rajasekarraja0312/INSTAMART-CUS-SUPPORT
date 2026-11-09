"""Part A: run sql/queries.sql against data/instamart.db, pandas cross-check, EDA."""
import re, sqlite3
import pandas as pd, numpy as np

con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql").read()
queries = {m.group(1): m.group(2).strip().rstrip(";")
           for m in re.finditer(r"-- name: (Q\d)\n(.*?;)", text, re.S)}
R = {k: pd.read_sql_query(q, con) for k, q in queries.items()}

print("=== SQL RESULTS ===")
print("Q1 customers:", len(R["Q1"]))
q2 = R["Q2"]
print("Q2 rows:", len(q2), "| total:", round(q2.revenue.sum(), 2))
print("Q2 largest cell:", q2.iloc[0].to_dict())
print("Q3 products above average price:", len(R["Q3"]))
print("Q4 customers with no completed order:", len(R["Q4"]))
print("Q5 customers above average:", len(R["Q5"]), "| average spend:", R["Q5"].avg_total.iloc[0])
q6 = R["Q6"]
print("Q6 top per city:")
print(q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows with RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()))
q7 = R["Q7"]
print("Q7 daily rows:", len(q7), "| last cumulative:", q7.cumulative_revenue.iloc[-1])

# ---------- pandas cross-check ----------
o = pd.read_sql_query("SELECT * FROM orders WHERE status='completed'", con)
c = pd.read_sql_query("SELECT * FROM customers", con)
p = pd.read_sql_query("SELECT * FROM products", con)
m = o.merge(c, on="customer_id").merge(p, on="product_id")
pv = pd.pivot_table(m, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print("\n=== PIVOT (margins=True) ===")
print(pv.round(2).to_string())
print("pivot grand total:", round(pv.loc["All", "All"], 2), "| SQL total:", round(q2.revenue.sum(), 2),
      "| equal:", abs(pv.loc["All", "All"] - q2.revenue.sum()) < 0.01)
agg = m.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                            count=("order_amount", "count")).round(2)
print("\nper-city sum/mean/count:\n", agg.to_string())

# ---------- EDA ----------
a = o["order_amount"]
q1, q3 = a.quantile(.25), a.quantile(.75); iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\n=== EDA order_amount (completed) ===")
print(f"mean {a.mean():.2f} median {a.median():.2f} skew {a.skew():.2f}")
print(f"Q1 {q1:.2f} Q3 {q3:.2f} lower fence {lo:.2f} upper fence {hi:.2f}")
print(f"outliers below: {(a < lo).sum()} above: {(a > hi).sum()}")
corr = o[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3).to_string())
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
best = pairs.abs().idxmax()
print("largest |corr| pair:", best, round(pairs[best], 3))
