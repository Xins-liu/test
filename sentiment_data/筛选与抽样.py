import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('weibo_senti_100k.csv', header=None, names=['label', 'review'],encoding="gbk")

# 按类别分组
class_0 = df[df['label'] == 0]
class_1 = df[df['label'] == 1]

# 确定每个类别需要抽取的数量
n_per_class = 5000  # 每个类别5000条，共10000条

# 检查每个类别的样本数量是否足够
n_class_0 = min(len(class_0), n_per_class)
n_class_1 = min(len(class_1), n_per_class)

# 如果某个类别不足，调整另一个类别的抽取数量
if n_class_0 < n_per_class or n_class_1 < n_per_class:
    total_needed = 10000
    n_class_0 = min(len(class_0), total_needed - n_class_1)
    n_class_1 = min(len(class_1), total_needed - n_class_0)

# 随机抽样
sample_0 = class_0.sample(n=n_class_0, random_state=42) if n_class_0 > 0 else pd.DataFrame()
sample_1 = class_1.sample(n=n_class_1, random_state=42) if n_class_1 > 0 else pd.DataFrame()

# 合并样本
sample_df = pd.concat([sample_0, sample_1], ignore_index=True)

# 打乱顺序（可选）
sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存到新文件
sample_df.to_csv('weibo_sentiment.csv', index=False, header=False)

# 输出统计信息
print(f"原始数据统计:")
print(f"  类别0: {len(class_0)}条")
print(f"  类别1: {len(class_1)}条")
print(f"\n抽取数据统计:")
print(f"  类别0: {len(sample_0)}条")
print(f"  类别1: {len(sample_1)}条")
print(f"  总计: {len(sample_df)}条")
print(f"\n已保存到sampled_data.csv")