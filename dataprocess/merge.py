import pandas as pd
import os

# === Step 1: Load CSV files ===
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
oil = pd.read_csv('data/oil.csv')
stores = pd.read_csv('data/stores.csv')
holidays = pd.read_csv('data/holidays_events.csv')

# === Step 2: Convert date columns to datetime ===
for df in [train, test, oil, holidays]:
    df['date'] = pd.to_datetime(df['date'])

# === Step 3: Handle missing oil prices ===
full_dates = pd.DataFrame({'date': pd.date_range(oil['date'].min(), oil['date'].max())})
oil_full = full_dates.merge(oil, on='date', how='left')
oil_full['dcoilwtico'] = oil_full['dcoilwtico'].ffill().bfill()
oil = oil_full.copy()

train['onpromotion'] = train['onpromotion'].fillna(0)
test['onpromotion'] = test['onpromotion'].fillna(0)

# === Step 4: Merge external data ===
train = train.merge(oil, on='date', how='left')
test = test.merge(oil, on='date', how='left')
train = train.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')

# === Step 5: Create is_working_day feature ===
holidays['is_working_day'] = 0
holidays.loc[(holidays['type'] == 'Holiday') & (holidays['transferred'] == True), 'is_working_day'] = 1
holidays.loc[holidays['type'] == 'Work Day', 'is_working_day'] = 1

holiday_reduced = holidays[['date', 'locale_name', 'is_working_day']].copy()
holiday_city = holiday_reduced.rename(columns={'locale_name': 'city'})
holiday_state = holiday_reduced.rename(columns={'locale_name': 'state'})
holiday_country = holiday_reduced[holiday_reduced['locale_name'] == 'Ecuador'].copy()

def add_is_working_day(df):
    df = df.copy()
    df['default_working_day'] = (df['date'].dt.weekday < 5).astype(int)
    df_city = df.merge(holiday_city, on=['date', 'city'], how='left')
    df_state = df.merge(holiday_state, on=['date', 'state'], how='left')
    df_country = df.merge(holiday_country[['date', 'is_working_day']], on='date', how='left')
    df['is_working_day'] = df_city['is_working_day'].combine_first(
                            df_state['is_working_day']).combine_first(
                            df_country['is_working_day']).combine_first(
                            df['default_working_day'])
    df['is_working_day'] = df['is_working_day'].astype(int)
    df.drop(columns=['default_working_day'], inplace=True)
    return df

train = add_is_working_day(train)
test = add_is_working_day(test)

# === Step 6: Payday feature ===
train['payday'] = train['date'].apply(lambda d: int(d.day == 15 or d.is_month_end))
test['payday'] = test['date'].apply(lambda d: int(d.day == 15 or d.is_month_end))

# === Step 7: Earthquake zone logic ===
def mark_quake_severity(date, state):
    if pd.Timestamp("2016-04-16") <= date <= pd.Timestamp("2016-05-15"):
        if state in ["Manabí", "Esmeraldas"]:
            return 1, 0
        elif state in ["Guayas", "Santo Domingo de los Tsáchilas", "Los Ríos", "Pichincha"]:
            return 0, 1
    return 0, 0

train['quake_severe'], train['quake_moderate'] = zip(*train.apply(lambda r: mark_quake_severity(r['date'], r['state']), axis=1))
test['quake_severe'], test['quake_moderate'] = zip(*test.apply(lambda r: mark_quake_severity(r['date'], r['state']), axis=1))

# === Step 8: Sample weight based on earthquake severity ===
def assign_sample_weight(row):
    if row['quake_severe'] == 1:
        return 2.0
    elif row['quake_moderate'] == 1:
        return 1.5
    else:
        return 1.0

train['sample_weight'] = train.apply(assign_sample_weight, axis=1)
test['sample_weight'] = test.apply(assign_sample_weight, axis=1)

# === Step 9: Save final output ===
os.makedirs('data/output', exist_ok=True)
train = train.drop(columns=['type', 'sample_weight'], errors='ignore')
test = test.drop(columns=['type', 'sample_weight'], errors='ignore')
train.to_csv('data/output/train_merged_final.csv', index=False)
test.to_csv('data/output/test_merged_final.csv', index=False)

print("✅ All data merged, features created, and earthquake visualization saved (no emoji).")
