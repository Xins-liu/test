import pandas as pd

def merge_and_deduplicate(
    labeled_csv_path: str,
    unlabeled_text_path: str,
    output_csv_path: str,
    text_column_labeled: str = "review",
    label_column: str = "label"
):
    """
    整合两个文件并去重：

    1. 读取第一个两列 CSV（有列名），将标签列中的 0 全部替换为 -1。
    2. 读取第二个文件（无列名，仅一列文本），为这些文本添加标签 0，然后纵向合并到第一个文件。
    3. 对合并后的数据按文本列去重，保留首次出现的记录。
    4. 将结果保存为 CSV。

    参数：
        labeled_csv_path: 第一个文件路径（两列 CSV，第一列标签，第二列文本）
        unlabeled_text_path: 第二个文件路径（无列名，仅一列文本）
        output_csv_path: 输出去重后 CSV 的保存路径
        text_column_labeled: 第一个文件中文本列的列名（默认为 "text"）
        label_column: 第一个文件中标签列的列名（默认为 "label"）
    """
    # ---------- 1. 读取第一个带标签的文件 ----------
    df1 = pd.read_csv(labeled_csv_path,encoding='gbk')

    # 确保列名正确；若第一个文件没有表头，需要先指定列名，这里假设有表头
    # 如果没有表头，可以用 pd.read_csv(labeled_csv_path, header=None, names=[label_column, text_column_labeled])
    # 根据你的实际情况调整，下面按有表头处理

    # 将标签列中的 0 替换为 -1
    df1[label_column] = df1[label_column].replace(0, -1)

    # ---------- 2. 读取第二个只有文本的文件 ----------
    # 假设第二个文件没有列名，只有一列文本，列名我们临时命名为 text_column_labeled
    df2 = pd.read_csv(unlabeled_text_path, header=None, names=[text_column_labeled])

    # 为这些文本添加标签列，全部赋值为 0
    df2[label_column] = 0

    # ---------- 3. 纵向合并两个 DataFrame ----------
    merged = pd.concat([df1, df2], ignore_index=True)

    # ---------- 4. 按文本列去重（保留第一次出现的记录） ----------
    merged_dedup = merged.drop_duplicates(subset=text_column_labeled, keep='first')

    # ---------- 5. 保存结果 ----------
    merged_dedup.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"整合并去重完成，共 {len(merged_dedup)} 条记录，已保存至 {output_csv_path}")

if __name__ == "__main__":
    # 修改为你的实际文件路径
    merge_and_deduplicate(
        labeled_csv_path="weibo_sentiment.csv",          # 第一个带标签的 CSV
        unlabeled_text_path="neutral.csv",       # 第二个纯文本文件（一列）
        output_csv_path="pos_nu_neg.csv"     # 输出文件
    )