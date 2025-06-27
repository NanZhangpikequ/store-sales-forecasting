import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Flatten
from sklearn.model_selection import train_test_split


# === 数据路径（Kaggle Working Dir） ===
data_path = '/kaggle/working/output/train_merged_final_with_features.csv'

# === 加载数据 ===
df = pd.read_csv(data_path)
df['sales_log'] = np.log1p(df['sales'])
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['store_nbr', 'family', 'date'])

# === 特征定义 ===
features = [
    'onpromotion_scaled', 'dcoilwtico_scaled',
    'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos',
    'is_working_day', 'quake_severe', 'quake_moderate', 'payday'
]
categorical = ['store_nbr_encoded', 'family_encoded', 'city_encoded']

# === 序列构造函数 ===
def create_sequences(df, seq_length=14, pred_length=14):
    X_seq, y_seq, cat_seq = [], [], []
    grouped = df.groupby(['store_nbr_encoded', 'family_encoded'])
    
    for (store, fam), group in grouped:
        group = group.reset_index(drop=True)
        
        data_x = group[features].values  # shape (T, feature_dim)
        data_y = group['sales_log'].values  # shape (T,)
        data_cat = group.loc[0, categorical].values  # 类别是固定的，取第一个就行
        
        total_len = len(group)
        max_start = total_len - seq_length - pred_length + 1
        if max_start <= 0:
            continue  # 数据不足，跳过
        
        # 利用滑窗视图创建输入序列 (shape: max_start, seq_length, feature_dim)
        x_windows = np.lib.stride_tricks.sliding_window_view(data_x, window_shape=seq_length, axis=0)[:max_start]
        
        # 预测目标序列 (shape: max_start, pred_length)
        y_windows = np.lib.stride_tricks.sliding_window_view(data_y, window_shape=pred_length, axis=0)[seq_length:seq_length+max_start]
        
        # 类别特征扩展
        cat_expanded = np.repeat(data_cat.reshape(1, -1), repeats=max_start, axis=0)
        
        X_seq.append(x_windows)
        y_seq.append(y_windows)
        cat_seq.append(cat_expanded)
    
    # 拼接所有组合的数据
    X_seq = np.concatenate(X_seq, axis=0)
    y_seq = np.concatenate(y_seq, axis=0)
    cat_seq = np.concatenate(cat_seq, axis=0)
    
    return X_seq, y_seq, cat_seq

# === 生成训练数据 ===
X, y, X_cat = create_sequences(df)
print(X.shape, y.shape, X_cat.shape)

# === 划分训练集与验证集 ===
X_train, X_val, y_train, y_val, X_cat_train, X_cat_val = train_test_split(
    X, y, X_cat, test_size=0.2, random_state=42
)

# === 修复数据类型，确保嵌入输入为 int32 ===
X_cat_train = X_cat_train.astype('int32')
X_cat_val = X_cat_val.astype('int32')

# === 构建模型并放置到 GPU 上 ===
with tf.device('/GPU:0'):
    # 输入层
    seq_input = Input(shape=(X.shape[1], X.shape[2]), name='seq_input')
    cat_store = Input(shape=(1,), dtype='int32', name='store_input')
    cat_family = Input(shape=(1,), dtype='int32', name='family_input')
    cat_city = Input(shape=(1,), dtype='int32', name='city_input')

    # 嵌入层
    emb_store = Embedding(input_dim=df['store_nbr_encoded'].nunique() + 1, output_dim=4)(cat_store)
    emb_family = Embedding(input_dim=df['family_encoded'].nunique() + 1, output_dim=4)(cat_family)
    emb_city = Embedding(input_dim=df['city_encoded'].nunique() + 1, output_dim=4)(cat_city)

    # 展平嵌入
    flat_store = Flatten()(emb_store)
    flat_family = Flatten()(emb_family)
    flat_city = Flatten()(emb_city)

    # 合并嵌入
    flat_cat = Concatenate()([flat_store, flat_family, flat_city])

    # LSTM 和输出
    lstm_out = LSTM(64)(seq_input)
    concat = Concatenate()([lstm_out, flat_cat])
    output = Dense(y.shape[1])(concat)

    # 模型构建与编译
    model = Model(inputs=[seq_input, cat_store, cat_family, cat_city], outputs=output)
    model.compile(optimizer='adam', loss='mse')

    # 开始训练
    print("\n=== 开始训练模型 ===")
    start_time = time.time()

    history = model.fit(
        [X_train, X_cat_train[:, 0], X_cat_train[:, 1], X_cat_train[:, 2]],
        y_train,
        validation_data=([X_val, X_cat_val[:, 0], X_cat_val[:, 1], X_cat_val[:, 2]], y_val),
        epochs=10,
        batch_size=64
    )

    end_time = time.time()
    print(f"✅ 训练完成，耗时 {end_time - start_time:.2f} 秒")

# === 绘制损失曲线 ===
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("/kaggle/working/lstm_loss_plot.png")
plt.show()
