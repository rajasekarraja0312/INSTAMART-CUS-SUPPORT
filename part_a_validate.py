"""Validate Part A reference checkpoints (tasks 3 and 4)."""
import pandas as pd
c = pd.read_csv("data/clean/customers.csv"); o = pd.read_csv("data/clean/orders.csv")
done = o[o.status == "completed"]
checks = {
 "cities counts": c.city.value_counts().to_dict() == {"Pune":227,"Delhi":210,"Bengaluru":197,"Chennai":191,"Hyderabad":191,"Mumbai":184},
 "six cities": c.city.nunique() == 6,
 "genders": c.gender.value_counts().to_dict() == {"female":610,"male":560,"unknown":30},
 "no missing ages": c.age.isna().sum() == 0,
 "ages 18-60": c.age.between(18, 60).all(),
 "12309 orders, unique ids": len(o) == 12309 and o.order_id.is_unique,
 "11110 completed / 1199 cancelled": (o.status == "completed").sum() == 11110 and (o.status == "cancelled").sum() == 1199,
 "no pending": (o.status == "pending").sum() == 0,
 "no missing amount": o.order_amount.isna().sum() == 0,
 "no missing delivery (completed)": done.delivery_minutes.isna().sum() == 0,
 "1199 missing delivery (cancelled)": o[o.status == "cancelled"].delivery_minutes.isna().sum() == 1199,
 "completed revenue 2,998,668.80": round(done.order_amount.sum(), 2) == 2998668.80,
}
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
assert all(checks.values()), "a checkpoint failed"
print("ALL CHECKPOINTS PASSED")
