import pandas as pd

# === Step 1: 加载 CSV 文件 ===
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
oil = pd.read_csv('data/oil.csv')
transactions = pd.read_csv('data/transactions.csv')
stores = pd.read_csv('data/stores.csv')
holidays = pd.read_csv('data/holidays_events.csv')

# === Step 2: 日期转换为 datetime 格式 ===
for df in [train, test, oil, transactions, holidays]:
    df['date'] = pd.to_datetime(df['date'])

# === Step 3: 处理缺失值 ===
oil['dcoilwtico'] = oil['dcoilwtico'].fillna(method='ffill')

if 'onpromotion' in train.columns:
    train['onpromotion'].fillna(0, inplace=True)
if 'onpromotion' in test.columns:
    test['onpromotion'].fillna(0, inplace=True)

# === Step 4: 合并表格 ===
train = train.merge(oil, on='date', how='left')
test = test.merge(oil, on='date', how='left')

train = train.merge(transactions, on=['date', 'store_nbr'], how='left')
test = test.merge(transactions, on=['date', 'store_nbr'], how='left')

train = train.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')

# === Step 5: is_working_day（节假日 vs 工作日） ===
holidays['is_working_day'] = 0
holidays.loc[holidays['type'] == 'Work Day', 'is_working_day'] = 1
holidays.loc[holidays['transferred'] == True, 'is_working_day'] = 1
working_day_dict = holidays.set_index('date')['is_working_day'].to_dict()

train['is_working_day'] = train['date'].map(working_day_dict).fillna(1).astype(int)
test['is_working_day'] = test['date'].map(working_day_dict).fillna(1).astype(int)

# === Step 6: earthquake_effect（2016-04-16 ~ 2016-05-15） ===
def mark_earthquake(date):
    return int(pd.Timestamp("2016-04-16") <= date <= pd.Timestamp("2016-05-15"))

train['earthquake_effect'] = train['date'].apply(mark_earthquake)
test['earthquake_effect'] = test['date'].apply(mark_earthquake)

# === Step 7: payday（每月15日与月末） ===
train['payday'] = train['date'].apply(lambda d: int(d.day == 15 or d.is_month_end))
test['payday'] = test['date'].apply(lambda d: int(d.day == 15 or d.is_month_end))

# === Step 8: 保存结果 ===
train.to_csv('data/output/train_merged_final.csv', index=False)
print("✅ train_merged_final.csv saved.")
test.to_csv('data/output/test_merged_final.csv', index=False)
print("✅ test_merged_final.csv saved.")

print("🎯 All merge and feature creation completed successfully.")
