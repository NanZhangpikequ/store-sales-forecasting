import pandas as pd

# read csv files
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
oil = pd.read_csv('data/oil.csv')
transactions = pd.read_csv('data/transactions.csv')
stores = pd.read_csv('data/stores.csv')
holidays = pd.read_csv('data/holidays_events.csv')

# ensure the type is datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
oil['date'] = pd.to_datetime(oil['date'])
transactions['date'] = pd.to_datetime(transactions['date'])
holidays['date'] = pd.to_datetime(holidays['date'])

# merge oil.csv
train = train.merge(oil, on='date', how='left')
test = test.merge(oil, on='date', how='left')
# merge transcations.csv
train = train.merge(transactions, on=['date', 'store_nbr'], how='left')
test = test.merge(transactions, on=['date', 'store_nbr'], how='left')
# merge store.csv
train = train.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')
# merge holidays_events.csv



# one-hot encoding?



# save the merged files
train.to_csv('data/output/train_merged.csv', index=False)
print("train.csv has been merged.")
test.to_csv('data/output/test_merged.csv', index=False)
print("test.csv has been merged.")

print("Merger completed")