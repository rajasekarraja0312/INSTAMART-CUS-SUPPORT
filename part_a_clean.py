"""Part A: audit raw data, clean it with documented rules, store CSV + SQLite."""
import os, re, sqlite3
import pandas as pd, numpy as np

RAW, CLEAN = "data/raw", "data/clean"
os.makedirs(CLEAN, exist_ok=True)

cust = pd.read_csv(f"{RAW}/customers.csv")
prod = pd.read_csv(f"{RAW}/products.csv")
orders = pd.read_csv(f"{RAW}/orders.csv")

# ---------------- AUDIT ----------------
print("=== AUDIT: shape, dtypes, missing values ===")
for name, df in [("customers", cust), ("products", prod), ("orders", orders)]:
    print(f"\n{name}: shape={df.shape}")
    summ = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                         "missing_pct": (df.isna().mean() * 100).round(2)})
    print(summ.to_string())

amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
audit = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount (53 'N/A' read as NaN by pandas + 67 'error')": int(amt_num.isna().sum()),
    "negative delivery_minutes": int((orders["delivery_minutes"] < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "blank age": int(cust["age"].isna().sum()),
    "age outside 10-100": int(((cust["age"] < 10) | (cust["age"] > 100)).sum()),
    "distinct raw city spellings": int(cust["city"].nunique()),
    "distinct non-blank raw gender spellings": int(cust["gender"].dropna().nunique()),
    "blank gender": int(cust["gender"].isna().sum()),
}
print("\n=== AUDIT FIGURES ===")
for k, v in audit.items():
    print(f"{k}: {v}")

# ---------------- CLEAN CUSTOMERS ----------------
c = cust.copy()
c["city"] = c["city"].str.strip().str.title()
gmap = {"m": "male", "male": "male", "f": "female", "female": "female"}
c["gender"] = c["gender"].fillna("").str.strip().str.lower().map(gmap).fillna("unknown")
n_bad_age = int(((c["age"] < 10) | (c["age"] > 100)).sum())
c.loc[(c["age"] < 10) | (c["age"] > 100), "age"] = np.nan
n_age_fill = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median")).astype(int)
c["signup_date"] = pd.to_datetime(c["signup_date"])
print("\n=== CLEAN CUSTOMERS ===")
print("cities:", c["city"].value_counts().to_dict())
print("genders:", c["gender"].value_counts().to_dict())
print(f"ages set to missing: {n_bad_age}; total ages filled with city median: {n_age_fill}")
print("missing ages:", int(c["age"].isna().sum()), "age range:", c["age"].min(), c["age"].max())

# ---------------- CLEAN ORDERS ----------------
o = orders.copy()
steps = {}
o["order_ts"] = pd.to_datetime(o["order_ts"])                                  # rule 1
n0 = len(o); o = o.drop_duplicates(); steps["exact duplicates removed"] = n0 - len(o)   # rule 2
n1 = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")   # rule 3
steps["older (pending) records removed, keep latest per order_id"] = n1 - len(o)
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")           # rule 4
o = o.merge(prod[["product_id", "category", "price_inr"]], on="product_id", how="left")
miss_amt = o["order_amount"].isna()
steps["order_amount filled from product price"] = int(miss_amt.sum())
o.loc[miss_amt, "order_amount"] = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)[miss_amt]
neg = o["delivery_minutes"] < 0                                                   # rule 5
steps["negative delivery_minutes set to missing"] = int(neg.sum())
o.loc[neg, "delivery_minutes"] = np.nan
comp = o["status"] == "completed"
med = o[comp].groupby("category")["delivery_minutes"].transform("median")
need = comp & o["delivery_minutes"].isna()
steps["completed delivery_minutes imputed (category median)"] = int(need.sum())
o.loc[need, "delivery_minutes"] = med[need]
o["delivery_minutes"] = o["delivery_minutes"].astype(float)
o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)

print("\n=== CLEAN ORDERS: rows changed per rule ===")
for k, v in steps.items():
    print(f"{k}: {v}")
print("rows:", len(o), "| status:", o["status"].value_counts().to_dict())
print("missing order_amount:", int(o["order_amount"].isna().sum()))
print("missing delivery completed:", int(o[o.status == 'completed']["delivery_minutes"].isna().sum()),
      "| cancelled:", int(o[o.status == 'cancelled']["delivery_minutes"].isna().sum()))
print("completed revenue INR:", round(o[o.status == 'completed']["order_amount"].sum(), 2))

# ---------------- STORE ----------------
c_out = c.copy(); c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
o_out = o.copy(); o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out.to_csv(f"{CLEAN}/customers.csv", index=False)
prod.to_csv(f"{CLEAN}/products.csv", index=False)
o_out.to_csv(f"{CLEAN}/orders.csv", index=False)
if os.path.exists("data/instamart.db"):
    os.remove("data/instamart.db")
with sqlite3.connect("data/instamart.db") as con:
    c_out.to_sql("customers", con, index=False)
    prod.to_sql("products", con, index=False)
    o_out.to_sql("orders", con, index=False)
print("\nstored: data/clean/*.csv and data/instamart.db (customers, products, orders)")
