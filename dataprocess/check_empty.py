import pandas as pd

def merge_consecutive_indices(indices):
    if not indices:
        return []
    indices = sorted(indices)
    ranges = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append((start, prev))
            start = prev = i
    ranges.append((start, prev))
    return ranges

def check_csv_null_with_date_ranges(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])  # 确保date是datetime类型

    has_null = df.isnull().values.any()
    print("是否存在空值：", has_null)
    if not has_null:
        print("恭喜！该CSV没有空值。")
        return
    
    print("\n每列空值数量：")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0])
    
    print("\n空值具体位置（按列，连续行合并区间，打印对应日期）：")
    for col in df.columns:
        null_rows = df.index[df[col].isnull()].tolist()
        if null_rows:
            ranges = merge_consecutive_indices(null_rows)
            # 对每个区间，取对应日期（区间行对应日期去重）
            date_strs = []
            for start, end in ranges:
                dates = df.loc[start:end, 'date'].dt.strftime('%Y-%m-%d').unique()
                # 一般连续空值的日期多是同一天，打印一次就行
                date_strs.append(", ".join(dates))
            print(f"列 '{col}' 存在空值，日期范围：{'; '.join(date_strs)}")

if __name__ == "__main__":
    choice = input("train(0) or test(1)? just input number: ")
    if choice == "0":
        choice = "train"
    elif choice == "1":
        choice = "test"
    else:
        print("just input 0 or 1.")
    file_path = f"data/output/{choice}_merged_final.csv"
    check_csv_null_with_date_ranges(file_path)
