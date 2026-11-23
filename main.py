import pandas as pd
import numpy as np
import math
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import os
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
import joblib
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
import warnings

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "4"


def round_two_decimal(values):
    if isinstance(values, (list, np.ndarray)):
        return np.round(np.array(values), 2)
    else:
        return round(float(values), 2)

plt.rcParams.update({
    "font.family": ["SimHei", "WenQuanYi Micro Hei", "Times New Roman"],
    "axes.unicode_minus": False,
    "font.size": 10
})
legend_font = FontProperties(family='Times New Roman', size=10)
tick_label_font = 'Times New Roman'
sns.set_style("white")

os.makedirs('results/stack_xgb_rf_mlp', exist_ok=True)
print("结果目录已创建：results/stack_xgb_rf_mlp")

def load_cof_data(data_path="filtered_cofs_results.csv", target="FE"):
    try:
        cof_df = pd.read_csv(data_path, encoding='gbk')
    except UnicodeDecodeError:
        cof_df = pd.read_csv(data_path, encoding='utf-8')
    cof_df = cof_df.dropna(subset=[target]).reset_index(drop=True)
    print(f"COF数据（目标变量：{target}）：{len(cof_df)}行 × {len(cof_df.columns)}列")
    
    all_features = [col for col in cof_df.columns if col != target]
    numerical_features = [col for col in all_features if pd.api.types.is_numeric_dtype(cof_df[col])]
    categorical_features = [col for col in all_features if not pd.api.types.is_numeric_dtype(cof_df[col])]
    
    print(f"\n特征分类：")
    print(f"- 数值特征：{len(numerical_features)}个 {numerical_features[:5]}...")
    print(f"- 分类特征：{len(categorical_features)}个 {categorical_features[:5]}...")
    print(f"- 总输入特征：{len(all_features)}个")
    return cof_df, target, numerical_features, categorical_features

cof_df, target_column, numerical_features, categorical_features = load_cof_data()
X = cof_df[[col for col in cof_df.columns if col != target_column]].copy()
y = cof_df[target_column].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=3408, shuffle=True
)
print(f"\nCOF数据划分：训练集{len(X_train)}条，测试集{len(X_test)}条（9:1）")


def create_preprocessor(numerical_features, categorical_features):
    transformers = []
    if numerical_features:
        transformers.append(('num', StandardScaler(), numerical_features))
    if categorical_features:
        transformers.append(
            ('cat',
             OneHotEncoder(handle_unknown='ignore'),
             categorical_features)
        )
    return ColumnTransformer(transformers=transformers)



def objective(trial):
    # XGBoost
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_nest', 100, 1000),
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('xgb_depth', 3, 10),
        'min_child_weight': trial.suggest_int('xgb_child', 1, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample', 0.5, 1.0),
        'gamma': trial.suggest_float('xgb_gamma', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('xgb_alpha', 1e-5, 1e2, log=True),
        'reg_lambda': trial.suggest_float('xgb_lambda', 1e-5, 1e2, log=True),
        'objective': 'reg:squarederror',
        'random_state': 3408,
        'n_jobs': -1,
        'verbosity': 0
    }

    # RandomForest
    rf_params = {
        'n_estimators': trial.suggest_int('rf_nest', 100, 1000),
        'max_depth': trial.suggest_int('rf_depth', 3, 20),
        'min_samples_split': trial.suggest_int('rf_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('rf_leaf', 1, 10),
        'max_features': trial.suggest_categorical('rf_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('rf_bootstrap', [True, False]),
        'random_state': 3408,
        'n_jobs': -1,
        'verbose': 0
    }

    # MLP meta
    mlp_meta_params = {
        'hidden_layer_sizes': (
            trial.suggest_int('meta_u1', 128, 512),
            trial.suggest_int('meta_u2', 64, 256),
            trial.suggest_int('meta_u3', 32, 128)
        ),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': trial.suggest_float('meta_alpha', 1e-6, 1e-3, log=True),
        'learning_rate_init': trial.suggest_float('meta_lr', 1e-4, 5e-3, log=True),
        'max_iter': 5000,
        'early_stopping': True,
        'validation_fraction': 0.2,
        'n_iter_no_change': 200,
        'random_state': 3408,
        'verbose': False
    }

    xgb_reg = xgb.XGBRegressor(**xgb_params)
    rf_reg = RandomForestRegressor(**rf_params)

    # Stacking
    stack = StackingRegressor(
        estimators=[('xgb', xgb_reg), ('rf', rf_reg)],
        final_estimator=MLPRegressor(**mlp_meta_params),
        cv=KFold(n_splits=3, shuffle=True, random_state=3408),
        passthrough=False,
        n_jobs=-1
    )

   
    preprocessor = create_preprocessor(numerical_features, categorical_features)
    pipeline = make_pipeline(preprocessor, stack)

    # 5折CV，完全隔离
    score = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    return -score.mean()


print("\n开始贝叶斯优化")
study = optuna.create_study(direction='minimize', study_name='stack_xgb_rf_mlp_onehot')
study.optimize(objective, n_trials=60, show_progress_bar=True)

best_mse = round_two_decimal(study.best_value)
best_rmse = round_two_decimal(np.sqrt(study.best_value))
print(f"\n最佳交叉验证 MSE={best_mse:.2f}, RMSE={best_rmse:.2f}")


bp = study.best_params


preprocessor = create_preprocessor(numerical_features, categorical_features)

xgb_best = xgb.XGBRegressor(
    n_estimators=bp['xgb_nest'],
    learning_rate=bp['xgb_lr'],
    max_depth=bp['xgb_depth'],
    min_child_weight=bp['xgb_child'],
    subsample=bp['xgb_subsample'],
    colsample_bytree=bp['xgb_colsample'],
    gamma=bp['xgb_gamma'],
    reg_alpha=bp['xgb_alpha'],
    reg_lambda=bp['xgb_lambda'],
    objective='reg:squarederror',
    random_state=3408, n_jobs=-1, verbosity=0
)

rf_best = RandomForestRegressor(
    n_estimators=bp['rf_nest'],
    max_depth=bp['rf_depth'],
    min_samples_split=bp['rf_split'],
    min_samples_leaf=bp['rf_leaf'],
    max_features=bp['rf_features'],
    bootstrap=bp['rf_bootstrap'],
    random_state=3408, n_jobs=-1
)

mlp_meta_best = MLPRegressor(
    hidden_layer_sizes=(bp['meta_u1'], bp['meta_u2'], bp['meta_u3']),
    alpha=bp['meta_alpha'],
    learning_rate_init=bp['meta_lr'],
    max_iter=5000, early_stopping=True, random_state=3408
)

stack_model = StackingRegressor(
    estimators=[('xgb', xgb_best), ('rf', rf_best)],
    final_estimator=mlp_meta_best,
    cv=KFold(3, shuffle=True, random_state=3408),
    passthrough=False, n_jobs=-1
)


final_model = make_pipeline(preprocessor, stack_model)
final_model.fit(X_train, y_train)


xgb_pipe = make_pipeline(create_preprocessor(numerical_features, categorical_features), xgb_best)
rf_pipe = make_pipeline(create_preprocessor(numerical_features, categorical_features), rf_best)
xgb_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)


y_train_pred_raw = final_model.predict(X_train)
y_test_pred_raw = final_model.predict(X_test)
xgb_test_pred = xgb_pipe.predict(X_test)
rf_test_pred = rf_pipe.predict(X_test)


train_r2 = round_two_decimal(r2_score(y_train, y_train_pred_raw))
test_r2 = round_two_decimal(r2_score(y_test, y_test_pred_raw))
train_rmse = round_two_decimal(np.sqrt(mean_squared_error(y_train, y_train_pred_raw)))
test_rmse = round_two_decimal(np.sqrt(mean_squared_error(y_test, y_test_pred_raw)))
train_mae = round_two_decimal(mean_absolute_error(y_train, y_train_pred_raw))
test_mae = round_two_decimal(mean_absolute_error(y_test, y_test_pred_raw))

xgb_r2 = round_two_decimal(r2_score(y_test, xgb_test_pred))
rf_r2 = round_two_decimal(r2_score(y_test, rf_test_pred))
best_base_r2 = max(xgb_r2, rf_r2)
gain = test_r2 - best_base_r2

print("\n" + "-"*60)
print(f"XGBoost 测试 R²: {xgb_r2:.4f}")
print(f"RF      测试 R²: {rf_r2:.4f}")
print(f"Stack   测试 R²: {test_r2:.4f}")
print(f"提升增益:        {gain:.4f}")
print("-"*60)

# ----------------------
# 保存
# ----------------------
pred_df = pd.DataFrame({
    'FE_true': np.concatenate([y_train, y_test]).round(2),
    'XGB_pred': np.concatenate([xgb_pipe.predict(X_train), xgb_test_pred]).round(2),
    'RF_pred': np.concatenate([rf_pipe.predict(X_train), rf_test_pred]).round(2),
    'Stack_pred': np.concatenate([y_train_pred_raw, y_test_pred_raw]).round(2),
    'dataset': ['train']*len(y_train) + ['test']*len(y_test)
})
pred_df.to_csv("results/stack_xgb_rf_mlp/predictions.csv", index=False, encoding='utf-8-sig')

joblib.dump(final_model, "results/stack_xgb_rf_mlp/final_stack_model.pkl")
joblib.dump(xgb_pipe, "results/stack_xgb_rf_mlp/xgb_model.pkl")
joblib.dump(rf_pipe, "results/stack_xgb_rf_mlp/rf_model.pkl")

# ----------------------
# 绘图
# ----------------------
def plot_scatter():
    plt.figure(figsize=(7,7))
    plt.scatter(y_test, xgb_test_pred, c='#ccc', s=12, alpha=0.4, label=f'XGB {xgb_r2:.4f}')
    plt.scatter(y_test, rf_test_pred, c='#aaa', s=12, alpha=0.4, label=f'RF {rf_r2:.4f}')
    plt.scatter(y_train, y_train_pred_raw, c='#2c3e50', s=25, alpha=0.7, label=f'Stack Train {train_r2:.4f}')
    plt.scatter(y_test, y_test_pred_raw, c='#e74c3c', s=40, alpha=0.9, label=f'Stack Test {test_r2:.4f}', edgecolor='white')
    
    lims = [min(y.min(), y_test_pred_raw.min()), max(y.max(), y_test_pred_raw.max())]
    plt.plot(lims, lims, c='#9b59b6', lw=2, label='Ideal')
    plt.text(lims[1]*0.7, lims[0]*1.1, f'Gain = {gain:.4f}', bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.xlabel('Actual FE')
    plt.ylabel('Predicted FE')
    plt.title('Stacking XGB+RF→MLP (OneHot, No Leakage)')
    plt.legend()
    plt.grid(ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/stack_xgb_rf_mlp/stack_plot.png", dpi=300)
    plt.close()

plot_scatter()


print("保存路径:", os.path.abspath("results/stack_xgb_rf_mlp"))
