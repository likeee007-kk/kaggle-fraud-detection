# Kaggle Playground S6E11: Credit Card Fraud Detection
# Best: 3xLGB + 2xCB Blending, Kaggle AUC 0.94737
# 实验记录见 README

import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier

# 1. 加载
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train = train.drop(columns=["id"])
test_ids = test["id"]
test = test.drop(columns=["id"])

# 2. 分离
y = train["is_fraud"]
X = train.drop(columns=["is_fraud"])

# 3. 目标编码
cat_cols = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
for col in cat_cols:
    fraud_rate = train.groupby(col)["is_fraud"].mean()
    X[col + "_fraud_rate"] = X[col].map(fraud_rate)
    test[col + "_fraud_rate"] = test[col].map(fraud_rate)

X = X.drop(columns=cat_cols)
test = test.drop(columns=cat_cols)

# 4. 布尔列
bool_cols = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
             "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
for col in bool_cols:
    X[col] = X[col].astype(int)
    test[col] = test[col].astype(int)

# 5. Blending: 3 LightGBM + 2 CatBoost
best_lgb = {
    "n_estimators": 410, "max_depth": 9,
    "learning_rate": 0.03151829548769726, "num_leaves": 26,
    "subsample": 0.8542553381697351, "colsample_bytree": 0.6352358465670511,
    "min_child_samples": 75,
}
best_cb = {
    "iterations": 767, "depth": 4,
    "learning_rate": 0.0993946004359052, "l2_leaf_reg": 2.902853972843268,
    "border_count": 212, "random_strength": 0.663618168193371,
}

test_preds = np.zeros(len(test_ids))
for seed in [42, 123, 456]:
    m = lgb.LGBMClassifier(**best_lgb, random_state=seed, verbose=-1)
    m.fit(X, y)
    test_preds += m.predict_proba(test)[:, 1] / 5

for seed in [789, 1024]:
    m = CatBoostClassifier(**best_cb, random_seed=seed, verbose=0, thread_count=-1)
    m.fit(X, y)
    test_preds += m.predict_proba(test)[:, 1] / 5

submission = pd.DataFrame({"id": test_ids, "is_fraud": test_preds})
submission.to_csv("submission.csv", index=False)
print("→ submission.csv | 3xLGB + 2xCB | Kaggle AUC 0.94737")