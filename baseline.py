import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

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

# 3.5 目标编码：把每个类别对应的平均欺诈率变成一个新列
cat_cols = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
for col in cat_cols:
    # 在训练集上算出每个类别的平均欺诈率
    train_grouped = train.groupby(col)["is_fraud"].mean()
    
    # 把算出来的欺诈率映射成新特征
    X[col + "_fraud_rate"] = X[col].map(train_grouped)
    # 测试集必须用训练集算出来的规则来映射，绝不能自己算（防数据泄露）
    test[col + "_fraud_rate"] = test[col].map(train_grouped)

# 4. 直接删掉原始的文本列，只保留刚刚生成的 _fraud_rate 目标编码列
X = X.drop(columns=cat_cols)
test = test.drop(columns=cat_cols)

# 5. 布尔列 → 0/1 
bool_cols = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
             "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
for col in bool_cols:
    X[col] = X[col].astype(int)
    test[col] = test[col].astype(int)

# 6. 划分训练/验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 训练模型
import lightgbm as lgb

model = lgb.LGBMClassifier(n_estimators=200, max_depth=8,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8, random_state=42, verbose=-1)
model.fit(X_train, y_train)

# 8. 验证
preds = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, preds)
print(f"Validation AUC: {auc:.4f}")

# 9. 预测测试集
test_preds = model.predict_proba(test)[:, 1]
submission = pd.DataFrame({"id": test_ids, "is_fraud": test_preds})
submission.to_csv("submission.csv", index=False)
print("→ submission.csv saved")