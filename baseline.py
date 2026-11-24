import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 1. 加载
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2. 丢掉 id
train = train.drop(columns=["id"])
test_ids = test["id"]
test = test.drop(columns=["id"])

# 3. 分离 X 和 y
y = train["is_fraud"]
X = train.drop(columns=["is_fraud"])

# 3.5 目标编码 + 保留原始列
cat_cols = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
train_grouped = {}
for col in cat_cols:
    train_grouped[col] = train.groupby(col)["is_fraud"].mean()
    X[col + "_fraud_rate"] = X[col].map(train_grouped[col])
    test[col + "_fraud_rate"] = test[col].map(train_grouped[col])
    # 原始列转 category（LightGBM/CatBoost 原生支持）
    X[col] = X[col].astype("category")
    test[col] = test[col].astype("category")

# 4. 保留原始类别列（不改审），跟 fraud_rate 并存
X = X.drop(columns=[])  # 不要删
test = test.drop(columns=[])  # 不要删

# 5. 布尔列 → 0/1
bool_cols = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
             "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
for col in bool_cols:
    X[col] = X[col].astype(int)
    test[col] = test[col].astype(int)

# 5.5 交互特征（33.9x lift 的组合）
X["cvv_ip"] = X["cvv_retry_count"] * X["ip_country_mismatch"]
X["merchantrisk_ip"] = X["merchant_risk_score"] * X["ip_country_mismatch"]
X["velocity_ip"] = X["velocity_score"] * X["ip_country_mismatch"]
X["cvv_merchantrisk"] = X["cvv_retry_count"] * X["merchant_risk_score"]

test["cvv_ip"] = test["cvv_retry_count"] * test["ip_country_mismatch"]
test["merchantrisk_ip"] = test["merchant_risk_score"] * test["ip_country_mismatch"]
test["velocity_ip"] = test["velocity_score"] * test["ip_country_mismatch"]
test["cvv_merchantrisk"] = test["cvv_retry_count"] * test["merchant_risk_score"]

# 比率特征（注释掉，AUC 反而不如只用 Optuna）
# eps = 1e-6
# X["amount_to_balance"] = X["amount_usd"] / (X["account_balance_usd"] + eps)
# X["risk_per_hour"] = X["merchant_risk_score"] / (X["time_of_day_hour"] + eps)
# test["amount_to_balance"] = test["amount_usd"] / (test["account_balance_usd"] + eps)
# test["risk_per_hour"] = test["merchant_risk_score"] / (test["time_of_day_hour"] + eps)

# Optuna 搜索 LightGBM 最优参数
def objective(trial):
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

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)
print(f"\n[LightGBM Optuna] Best AUC: {study.best_value:.4f}")
print(f"[LightGBM Optuna] Best params: {study.best_params}")

best_lgb = study.best_params

# 6. CatBoost（固定最优参数）
from catboost import CatBoostClassifier

# 7. Blending: 3 LightGBM + 2 CatBoost + Pseudo-Labeling

best_cb = {
    "iterations": 767,
    "depth": 4,
    "learning_rate": 0.0993946004359052,
    "l2_leaf_reg": 2.902853972843268,
    "border_count": 212,
    "random_strength": 0.663618168193371,
}

N = len(test_ids)

# --- Round 1: 正常 Blending，拿到 test 预测 ---
test_preds_r1 = np.zeros(N)

for seed in [42, 123, 456]:
    m = lgb.LGBMClassifier(**best_lgb, random_state=seed, verbose=-1)
    m.fit(X, y)
    test_preds_r1 += m.predict_proba(test)[:, 1] * (1 / 5)

for seed in [789, 1024]:
    m = CatBoostClassifier(**best_cb, random_seed=seed, verbose=0, thread_count=-1)
    m.fit(X, y)
    test_preds_r1 += m.predict_proba(test)[:, 1] * (1 / 5)

# --- Pseudo-Labeling ---
# 挑最自信的 5000 条非欺诈（预测最低）+ 5000 条欺诈（预测最高）
confident_nonfraud = np.argsort(test_preds_r1)[:5000]   # 最低的 5000 条
confident_fraud = np.argsort(test_preds_r1)[-5000:]    # 最高的 5000 条
pseudo_indices = np.concatenate([confident_nonfraud, confident_fraud])

# 把这些高置信样本的预测值当作"真标签"加入训练集
X_pseudo = test.iloc[pseudo_indices].copy()
y_pseudo = (test_preds_r1[pseudo_indices] > 0.5).astype(int)

X2 = pd.concat([X, X_pseudo], ignore_index=True)
y2 = pd.concat([y, pd.Series(y_pseudo, name="is_fraud")], ignore_index=True)
print(f"[Pseudo] 训练集 {len(X)} + 伪标签 {len(X_pseudo)} = {len(X2)} 行")

# --- Round 2: 用增强后的训练集重新 Blending ---
test_preds_r2 = np.zeros(N)

for seed in [42, 123, 456]:
    m = lgb.LGBMClassifier(**best_lgb, random_state=seed, verbose=-1)
    m.fit(X2, y2)
    test_preds_r2 += m.predict_proba(test)[:, 1] * (1 / 5)

for seed in [789, 1024]:
    m = CatBoostClassifier(**best_cb, random_seed=seed, verbose=0, thread_count=-1)
    m.fit(X2, y2)
    test_preds_r2 += m.predict_proba(test)[:, 1] * (1 / 5)

submission = pd.DataFrame({"id": test_ids, "is_fraud": test_preds_r2})
submission.to_csv("submission.csv", index=False)
print("→ submission.csv saved (Blending + Pseudo-Labeling)")
