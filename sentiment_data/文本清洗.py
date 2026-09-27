import pandas as pd
import re


def simple_clean_keep_emotions(file_path):
    """
    简化的清洗函数：只保留表情符号和中文内容
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 获取第二列（review列）
    review_col = df.columns[1]

    # 备份原始数据
    df['original_review'] = df[review_col]

    def clean_text(text):
        if pd.isna(text):
            return text

        # 提取表情符号
        emotions = re.findall(r'\[[^\]]*\]', text)

        # 提取中文内容
        chinese = re.findall(r'[\u4e00-\u9fa5，。！？、；："'']+', text)

        # 组合结果
        result_parts = []
        if chinese:
            result_parts.append(' '.join(chinese))
        if emotions:
            result_parts.append(' '.join(emotions))

        return ' '.join(result_parts).strip()

    # 应用清洗
    df[review_col] = df[review_col].apply(clean_text)

    # 保存结果（覆盖原文件）
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

    print("清洗完成！结果示例:")
    for i in range(min(3, len(df))):
        print(f"原始: {df['original_review'].iloc[i]}")
        print(f"清洗后: {df[review_col].iloc[i]}")
        print("-" * 50)

    return df

# 使用示例
df = simple_clean_keep_emotions("weibo_senti_100k.csv")