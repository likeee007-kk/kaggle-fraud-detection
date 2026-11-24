import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna

# 1. 加载数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2. 处理id列
train = train.drop(columns=["id"])
test_ids = test["id"]
test = test.drop(columns=["id"])

# 3. 划分特征与标签
y = train["is_fraud"]
X = train.drop(columns=["is_fraud"])

# 3.5 目标编码 + 保留原始类别列
cat_cols = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
for col in cat_cols:
    fraud_rate = train.groupby(col)["is_fraud"].mean()
    X[col + "_fraud_rate"] = X[col].map(fraud_rate)
    test[col + "_fraud_rate"] = test[col].map(fraud_rate)
    X[col] = X[col].astype("category")
    test[col] = test[col].astype("category")

# 4. 布尔特征转为0/1
bool_cols = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"
]
for col in bool_cols:
    X[col] = X[col].astype(int)
    test[col] = test[col].astype(int)

# 5. 构造交互特征
X["cvv_ip"] = X["cvv_retry_count"] * X["ip_country_mismatch"]
X["merchantrisk_ip"] = X["merchant_risk_score"] * X["ip_country_mismatch"]
X["velocity_ip"] = X["velocity_score"] * X["ip_country_mismatch"]
X["cvv_merchantrisk"] = X["cvv_retry_count"] * X["merchant_risk_score"]

test["cvv_ip"] = test["cvv_retry_count"] * test["ip_country_mismatch"]
test["merchantrisk_ip"] = test["merchant_risk_score"] * test["ip_country_mismatch"]
test["velocity_ip"] = test["velocity_score"] * test["ip_country_mismatch"]
test["cvv_merchantrisk"] = test["cvv_retry_count"] * test["merchant_risk_score"]

# 6. LightGBM Optuna调参
def objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 4, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 16, 127),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "random_state": 42,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print("=== LightGBM Optuna (50 trials) ===")
study = optuna.create_study(direction="maximize")
study.optimize(objective_lgb, n_trials=50, show_progress_bar=True)
best_lgb = study.best_params
print(f"LGB Best AUC: {study.best_value:.4f}")
print(f"LGB Best params: {best_lgb}")

# 7. Blending: 3×LGB + 2×CatBoost（无伪标签）
best_cb = {
    "iterations": 767,
    "depth": 4,
    "learning_rate": 0.0993946004359052,
    "l2_leaf_reg": 2.902853972843268,
    "border_count": 212,
    "random_strength": 0.663618168193371,
}

test_preds = np.zeros(len(test_ids))

for seed in [42, 123, 456]:
    m = lgb.LGBMClassifier(**best_lgb, random_state=seed, verbose=-1)
    m.fit(X, y)
    test_preds += m.predict_proba(test)[:, 1] * (1 / 5)

for seed in [789, 1024]:
    m = CatBoostClassifier(**best_cb, random_seed=seed, verbose=0, thread_count=-1, cat_features=cat_cols)
    m.fit(X, y)
    test_preds += m.predict_proba(test)[:, 1] * (1 / 5)

submission = pd.DataFrame({"id": test_ids, "is_fraud": test_preds})
submission.to_csv("submission.csv", index=False)
print(f"→ submission.csv saved | LGB CV: {study.best_value:.4f}")
