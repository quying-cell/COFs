import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import warnings
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 28
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24

# 加载模型
model_path = "results/stack_xgb_rf_mlp/final_stack_model.pkl"
final_model = joblib.load(model_path)
print("Model loaded successfully")

# 加载数据
def load_cof_data(data_path="filtered_cofs_results.csv", target="FE"):
    try:
        df = pd.read_csv(data_path, encoding='gbk')
    except:
        df = pd.read_csv(data_path, encoding='utf-8')
    df = df.dropna(subset=[target]).reset_index(drop=True)
    return df

df = load_cof_data()
target = "FE"
X = df.drop(columns=[target])
y = df[target]

# 划分数据集（保持随机状态）
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=3408, shuffle=True
)

# 拆分管道
preprocessor = final_model.named_steps['columntransformer']
stack_model = final_model.named_steps['stackingregressor']

# 提取特征类型
num_features = [f for f in preprocessor.feature_names_in_ if f in preprocessor.transformers_[0][2]]
cat_features = preprocessor.transformers_[1][2]
onehot_encoder = preprocessor.transformers_[1][1]

# 获取OneHot编码后的特征名
def get_onehot_feature_names(encoder, cat_features):
    feature_names = []
    for i, cat in enumerate(cat_features):
        categories = encoder.categories_[i][1:]
        for val in categories:
            feature_names.append(f"{cat}_{val}")
    return feature_names

onehot_names = get_onehot_feature_names(onehot_encoder, cat_features)
all_processed_names = num_features + onehot_names
print("Processed features:", len(all_processed_names))
print("Numerical:", len(num_features), "| Categorical (OneHot):", len(onehot_names))

# 预处理
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 模型预测函数
def model_predict(X):
    return stack_model.predict(X)

print("\nComputing SHAP values (KernelExplainer)...")

# 背景数据
background = shap.sample(X_train_processed, 50)
explainer = shap.KernelExplainer(model_predict, background)

# 计算SHAP值（测试集前100条）
shap_values = explainer.shap_values(X_test_processed[:100])

# 合并OneHot特征的SHAP值
def merge_onehot_shap(shap_values, num_features, cat_features, onehot_encoder):
    n_samples = shap_values.shape[0]
    merged_names = num_features.copy()
    merged_names.extend(cat_features)
    n_merged = len(merged_names)
    merged_sv = np.zeros((n_samples, n_merged))

    n_num = len(num_features)
    merged_sv[:, :n_num] = shap_values[:, :n_num]

    cat_idx = n_num
    for i, cat in enumerate(cat_features):
        n_cats = len(onehot_encoder.categories_[i]) - 1
        sv_slice = shap_values[:, cat_idx : cat_idx + n_cats]
        merged_sv[:, n_num + i] = np.sum(sv_slice, axis=1)
        cat_idx += n_cats

    return merged_sv, merged_names

merged_shap_values, merged_feature_names = merge_onehot_shap(
    shap_values, num_features, cat_features, onehot_encoder
)

# ===================== 最终修复：确保所有特征都有颜色 =====================
X_test_raw = X_test.iloc[:100].copy()
feature_values = []

for feat in merged_feature_names:
    vals = X_test_raw[feat].values
    # 关键：把分类特征转成数字，让SHAP正常上色
    try:
        vals = vals.astype(float)
    except:
        unique_vals = np.unique(vals)
        val_map = {v: i for i, v in enumerate(unique_vals)}
        vals = np.array([val_map[v] for v in vals])
    feature_values.append(vals)

feature_values = np.column_stack(feature_values)

# 创建保存目录
os.makedirs('results/stack_xgb_rf_mlp/shap', exist_ok=True)

# ======================================================================
# 1. 【关键修改】标准默认蓝红配色 SHAP 蜂群图 (删除 cmap="coolwarm")
# ======================================================================
plt.figure(figsize=(14, 10))
shap.summary_plot(
    merged_shap_values,
    features=feature_values,
    feature_names=merged_feature_names,
    max_display=10,
    show=False
    # 这里不指定 cmap，使用 SHAP 默认的 标准蓝红
)
plt.xlabel("SHAP value", fontsize=24, fontfamily="Times New Roman")
plt.tight_layout()
plt.savefig("results/stack_xgb_rf_mlp/shap/shap_summary_color.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. 特征重要性柱状图
plt.figure(figsize=(14, 8))
shap.summary_plot(
    merged_shap_values,
    feature_names=merged_feature_names,
    plot_type="bar",
    max_display=10,
    show=False
)
plt.xlabel("SHAP value (mean|SHAP|)", fontsize=24, fontfamily="Times New Roman")
plt.tight_layout()
plt.savefig("results/stack_xgb_rf_mlp/shap/shap_importance.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nAll done! 已生成标准蓝红配色 SHAP 图！")  