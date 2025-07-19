# predict_lstm.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# === 特征定义，与训练脚本保持一致 ===
features = [
    'onpromotion_scaled', 'dcoilwtico_scaled',
    'month_sin', 'month_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'is_working_day', 'quake_severe',
    'quake_moderate', 'payday'
]
categorical = ['store_nbr_encoded', 'family_encoded', 'city_encoded']

# === 构造测试序列 ===
def create_test_sequences(df, seq_length=14):
    X_seq, cat_seq, keys, dates = [], [], [], []
    grouped = df.groupby(['store_nbr_encoded','family_encoded'])
    for (store,fam), g in grouped:
        g = g.sort_values('date').reset_index(drop=True)
        if len(g) < seq_length:
            continue
        X_seq.append(g[features].values[-seq_length:])
        cat_seq.append(g.loc[0, categorical].values.astype('int32'))
        keys.append((store, fam))
        dates.append(g['date'].values[-seq_length:])
    return np.array(X_seq), np.array(cat_seq), keys, dates

# === 加载测试集 ===
test_path = '/kaggle/input/after-merge-and-process/test_merged_final_with_features.csv'
df_test   = pd.read_csv(test_path)
df_test['date'] = pd.to_datetime(df_test['date'])

X_test, Xc_test, keys, date_list = create_test_sequences(df_test)

# === 加载最优模型 & 预测 ===
model = load_model('best_model.h5', compile=False)
pred_log = model.predict(
    [X_test, Xc_test[:,0], Xc_test[:,1], Xc_test[:,2]],
    verbose=1
)
pred_sales = np.expm1(pred_log)

# === 展开为长表并保存 ===
rows = []
for (store,fam), dates, sales in zip(keys, date_list, pred_sales):
    for dt, s in zip(dates, sales):
        rows.append({
            'store_nbr_encoded': store,
            'family_encoded': fam,
            'date': dt,
            'pred_sales': float(s)
        })

pred_df = pd.DataFrame(rows)
output_path = '/kaggle/working/output/predicted_sales.csv'
pred_df.to_csv(output_path, index=False)
print(f"✅ 测试集预测完成，结果保存在：{output_path}")
