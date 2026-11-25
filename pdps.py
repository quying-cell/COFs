import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import joblib, os, csv, warnings
from tqdm import tqdm

# 关闭警告提示
warnings.filterwarnings('ignore')

# ===================== 全局配置（字体完全统一） =====================
plt.rcParams.update({
    "font.family": ["Times New Roman"],
    "axes.unicode_minus": False,
    "font.size": 32,
    "axes.labelsize": 32,
    "legend.fontsize": 32,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
    "axes.facecolor": 'white',
    "grid.color": '#EEEEEE',
    "grid.linestyle": '--',
    "grid.linewidth": 0.8,
    "axes.edgecolor": '#333333',
    "xtick.color": '#333333',
    "ytick.color": '#333333',
    "axes.labelcolor": '#333333',
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1,
    "ytick.major.width": 1
})
sns.set(font='Times New Roman')

font = FontProperties(
    family='Times New Roman',
    size=32,
    weight='normal'
)

output_dir = 'results/feature_pdp_ci_fast'
os.makedirs(output_dir, exist_ok=True)
print(f'结果保存路径：{output_dir}\n')

# 正确路径：只加载最终完整模型
model_dir = "results/stack_xgb_rf_mlp"
final_model_path = f"{model_dir}/final_stack_model.pkl"

target_features = ['UT', 'TF', 'PS', 'CT', 'PLD', "BET", "J", "pH", "PT"]
BOOTSTRAP_N = 30
X_POINTS = 30

# ===================== 工具函数 =====================
def get_sep(file_path, enc='gbk'):
    with open(file_path, 'r', encoding=enc, errors='ignore') as f:
        return csv.Sniffer().sniff(f.read(10000)).delimiter

def search_data_file():
    print('搜索数据文件...')
    target_files = ['filtered_cofs_results.csv', 'cofs.csv', 'cofs.txt']
    for fname in target_files:
        if os.path.exists(fname):
            print(f'发现数据文件：{fname}')
            return fname
    while True:
        path = input('请手动输入数据文件路径：').strip()
        if os.path.exists(path):
            return path
        print('路径无效，请重新输入！')

# ===================== 资源加载函数 =====================
def load_all_resources():
    if not os.path.exists(final_model_path):
        raise FileNotFoundError(f'缺失模型文件：{final_model_path}')
    
    print('加载完整堆叠模型管道...')
    full_pipeline = joblib.load(final_model_path)
    
    # 从完整管道中提取
    preprocessor = full_pipeline.named_steps['columntransformer']
    stack_model = full_pipeline.named_steps['stackingregressor']
    
    data_path = search_data_file()
    for enc in ['gbk', 'utf-8', 'gb2312', 'latin-1']:
        try:
            df = pd.read_csv(
                data_path, encoding=enc, sep=get_sep(data_path, enc),
                quotechar='"', on_bad_lines='skip', engine='python'
            )
            break
        except:
            continue
    if 'df' not in locals():
        raise ValueError('数据文件读取失败，尝试多种编码均无效')
    
    df.columns = [col.strip().replace('\u200b', '').replace('\xa0', '').replace(' ', '') 
                  for col in df.columns]
    
    feature_names = []
    for _, _, cols in preprocessor.transformers_:
        feature_names.extend(cols)
    
    valid_feats = []
    non_numeric_feats = []
    for feat in target_features:
        if feat in feature_names:
            if pd.api.types.is_numeric_dtype(df[feat]):
                valid_feats.append(feat)
            else:
                non_numeric_feats.append(feat)
    
    if non_numeric_feats:
        print(f"警告：非数值特征已跳过：{non_numeric_feats}")
    invalid = set(target_features) - set(valid_feats) - set(non_numeric_feats)
    if invalid:
        print(f'过滤无效特征：{invalid}')
    if not valid_feats:
        raise ValueError('无可用特征进行分析')
    
    for col in feature_names:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"特征 '{col}' 用中位数 {median_val:.4f} 填充。")
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(f'数据处理完成 | 有效样本：{len(df)} | 分析特征：{valid_feats}\n')
    return df, valid_feats, feature_names, stack_model, preprocessor

# ===================== 核心PDP函数（已修复稀疏矩阵报错） =====================
def generate_pdp_with_ci_fast(df, feat_name, model, preprocessor, all_feature_names):
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=1200)
    
    X_pdp = df[all_feature_names].copy()
    x_min, x_max = df[feat_name].min(), df[feat_name].max()
    x_pdp_vals = np.linspace(x_min, x_max, X_POINTS)
    
    pdp_mean = []
    feat_idx = all_feature_names.index(feat_name)
    X_processed_base = preprocessor.transform(X_pdp)
    
    for val in x_pdp_vals:
        X_pdp_temp = X_pdp.copy()
        X_pdp_temp[feat_name] = val
        X_processed_temp = preprocessor.transform(X_pdp_temp)
        
        # 直接使用，不再手动reshape+flatten，避免稀疏矩阵报错
        pred = model.predict(X_processed_temp)
        pdp_mean.append(np.mean(pred))
    pdp_mean = np.array(pdp_mean)
    
    print(f'【{feat_name}】开始Bootstrap采样（{BOOTSTRAP_N}次）...')
    bootstrap_pdp = []
    for i in tqdm(range(BOOTSTRAP_N), desc=f'  {feat_name}采样进度'):
        sample_idx = np.random.choice(len(df), size=len(df), replace=True)
        X_sample = df[all_feature_names].iloc[sample_idx].copy()
        
        pdp_sample = []
        for val in x_pdp_vals:
            X_pdp_temp = X_sample.copy()
            X_pdp_temp[feat_name] = val
            X_processed_temp = preprocessor.transform(X_pdp_temp)
            
            pred = model.predict(X_processed_temp)
            pdp_sample.append(np.mean(pred))
        bootstrap_pdp.append(pdp_sample)
    
    bootstrap_pdp = np.array(bootstrap_pdp)
    pdp_lower = np.percentile(bootstrap_pdp, 2.5, axis=0)
    pdp_upper = np.percentile(bootstrap_pdp, 97.5, axis=0)
    
    ax1.plot(x_pdp_vals, pdp_mean, color='#2C7FB8', linewidth=3, alpha=0.9, label='PDP')
    ax1.fill_between(x_pdp_vals, pdp_lower, pdp_upper, color='#2C7FB8', alpha=0.2, label='95% CI')
    
    ax1.set_xlabel(f'{feat_name}', fontproperties=font, labelpad=10)
    ax1.set_ylabel('Partial Dependence', fontproperties=font, labelpad=10)
    ax1.tick_params(axis='both', labelsize=32)
    ax1.grid(color='#EEEEEE', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(prop=font, loc='best')

    # X轴：6个整数刻度
    x_ticks = np.linspace(x_min, x_max, 6).astype(int)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([str(tick) for tick in x_ticks])
    
    # Y轴：6个整数刻度
    y_min, y_max = ax1.get_ylim()
    y_ticks = np.linspace(y_min, y_max, 6).astype(int)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels([str(tick) for tick in y_ticks])

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{feat_name}_pdp_ci_fast.png')
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close()
    print(f'【{feat_name}】PDP+95%置信区间图已保存：{save_path}\n')

# ===================== 主执行 =====================
if __name__ == '__main__':
    try:
        print('='*60)
        print('快速版PDP + 95%置信区间分析（字体统一：Times New Roman 32号）')
        print('='*60)
        
        data, valid_features, all_feature_names, model, preprocessor = load_all_resources()
        
        print('开始生成PDP+95%置信区间分析图...\n')
        for feat in valid_features:
            generate_pdp_with_ci_fast(data, feat, model, preprocessor, all_feature_names)
        
        print('='*60)
        print(f'所有分析任务完成！结果文件位于：{output_dir}')
        print('='*60)
    
    except Exception as e:
        print(f'\n程序执行异常：{str(e)}')