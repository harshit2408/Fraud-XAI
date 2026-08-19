import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

print('Loading data...')
X_train = pd.read_parquet('data/processed/train_features.parquet')
y_train = pd.read_parquet('data/processed/train_labels.parquet').squeeze()
X_val = pd.read_parquet('data/processed/val_features.parquet')
y_val = pd.read_parquet('data/processed/val_labels.parquet').squeeze()
X_test = pd.read_parquet('data/processed/test_features.parquet')
y_test = pd.read_parquet('data/processed/test_labels.parquet').squeeze()

X_train_full = pd.concat([X_train, X_val], ignore_index=True)
y_train_full = pd.concat([y_train, y_val], ignore_index=True)

print('Training XGBoost with TEST set for early stopping (to check for leakage)...')
model = xgb.XGBClassifier(
    n_estimators=1000, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(y_train_full == 0).sum() / (y_train_full == 1).sum(),
    eval_metric='aucpr', early_stopping_rounds=100, random_state=42,
    enable_categorical=True, n_jobs=-1
)

model.fit(X_train_full, y_train_full, eval_set=[(X_test, y_test)], verbose=100)

print(f'Best Test PR-AUC achieved (leaked): {model.best_score}')
