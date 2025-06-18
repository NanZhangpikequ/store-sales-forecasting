import pandas as pd

# === Step 1: 加载 CSV 文件 ===
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
oil = pd.read_csv('data/oil.csv')
stores = pd.read_csv('data/stores.csv')
holidays = pd.read_csv('data/holidays_events.csv')

# === Step 2: 日期转换为 datetime 格式 ===
for df in [train, test, oil, holidays]:
    df['date'] = pd.to_datetime(df['date'])

# === Step 3: 处理缺失值 ===
oil['dcoilwtico'] = oil['dcoilwtico'].ffill()

if 'onpromotion' in train.columns:
    train['onpromotion'] = train['onpromotion'].fillna(0)
if 'onpromotion' in test.columns:
    test['onpromotion'] = test['onpromotion'].fillna(0)

# === Step 4: 合并表格 ===
train = train.merge(oil, on='date', how='left')
test = test.merge(oil, on='date', how='left')

train = train.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')

# === Step 5: is_working_day（节假日 vs 工作日） ===

# 1. 先给 holidays 加上 is_working_day 字段，标记这天是否是工作日
holidays['is_working_day'] = 0
holidays.loc[(holidays['type'] == 'Holiday') & (holidays['transferred'] == True), 'is_working_day'] = 1
holidays.loc[holidays['type'] == 'Work Day', 'is_working_day'] = 1

# 2. 准备三个节假日映射表，分别对应 city、state 和全国（Ecuador）
holiday_reduced = holidays[['date', 'locale_name', 'is_working_day']].copy()
holiday_city = holiday_reduced.rename(columns={'locale_name': 'city'})
holiday_state = holiday_reduced.rename(columns={'locale_name': 'state'})
holiday_country = holiday_reduced[holiday_reduced['locale_name'] == 'Ecuador'].copy()

# 3. 定义一个处理函数，给任意数据集添加is_working_day字段
def add_is_working_day(df):
    df = df.copy()
    df['default_working_day'] = (df['date'].dt.weekday < 5).astype(int)  # 默认周一到周五工作

    # merge 查找 city 对应的节假日工作状态
    df_city = df.merge(holiday_city, on=['date', 'city'], how='left')
    # merge 查找 state 对应的节假日工作状态
    df_state = df.merge(holiday_state, on=['date', 'state'], how='left')
    # merge 查找全国对应的节假日工作状态
    df_country = df.merge(holiday_country[['date', 'is_working_day']], on='date', how='left')

    # 按优先级合并：city > state > country > 默认
    df['is_working_day'] = df_city['is_working_day'].combine_first(
                            df_state['is_working_day']).combine_first(
                            df_country['is_working_day']).combine_first(
                            df['default_working_day'])

    # 转成int类型（0或1）
    df['is_working_day'] = df['is_working_day'].astype(int)

    # 可以删掉中间列，保持整洁
    df.drop(columns=['default_working_day'], inplace=True)

    return df

# 4. 使用函数处理train和test
train = add_is_working_day(train)
test = add_is_working_day(test)

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
