import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# === Load data ===
df = pd.read_csv("train_merged_final_with_features.csv")
df['sales_log'] = np.log1p(df['sales'])
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['store_nbr', 'family', 'date'])

# === Feature definitions ===
features = [
    'onpromotion_scaled', 'dcoilwtico_scaled',
    'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos',
    'is_working_day', 'quake_severe', 'quake_moderate', 'payday'
]
categorical = ['store_nbr_encoded', 'family_encoded', 'city_encoded']

# === Create sequences ===
def create_sequences(df, seq_length=14, pred_length=14):
    X_seq, y_seq, cat_seq = [], [], []
    for (store, fam), group in df.groupby(['store_nbr_encoded', 'family_encoded']):
        group = group.reset_index(drop=True)
        for i in range(len(group) - seq_length - pred_length):
            x = group.loc[i:i+seq_length-1, features].values
            y = group.loc[i+seq_length:i+seq_length+pred_length-1, 'sales_log'].values
            c = group.loc[i, categorical].values
            X_seq.append(x)
            y_seq.append(y)
            cat_seq.append(c)
    return np.array(X_seq), np.array(y_seq), np.array(cat_seq)

X, y, X_cat = create_sequences(df)

# === Train/Val split ===
X_train, X_val, y_train, y_val, X_cat_train, X_cat_val = train_test_split(
    X, y, X_cat, test_size=0.2, random_state=42)

# === Model Inputs ===
seq_input = Input(shape=(X.shape[1], X.shape[2]), name='seq_input')
cat_store = Input(shape=(1,), name='store_input')
cat_family = Input(shape=(1,), name='family_input')
cat_city = Input(shape=(1,), name='city_input')

# === Embeddings ===
emb_store = Embedding(input_dim=df['store_nbr_encoded'].nunique()+1, output_dim=4)(cat_store)
emb_family = Embedding(input_dim=df['family_encoded'].nunique()+1, output_dim=4)(cat_family)
emb_city = Embedding(input_dim=df['city_encoded'].nunique()+1, output_dim=4)(cat_city)

flat_cat = Concatenate()([
    tf.squeeze(emb_store, axis=1),
    tf.squeeze(emb_family, axis=1),
    tf.squeeze(emb_city, axis=1)
])

# === LSTM Model ===
lstm_out = LSTM(64)(seq_input)
concat = Concatenate()([lstm_out, flat_cat])
output = Dense(y.shape[1])(concat)

model = Model(inputs=[seq_input, cat_store, cat_family, cat_city], outputs=output)
model.compile(optimizer='adam', loss='mse')

# === Training ===
history = model.fit(
    [X_train, X_cat_train[:, 0], X_cat_train[:, 1], X_cat_train[:, 2]],
    y_train,
    validation_data=([X_val, X_cat_val[:, 0], X_cat_val[:, 1], X_cat_val[:, 2]], y_val),
    epochs=10,
    batch_size=64
)

# === Visualization ===
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lstm_loss_plot.png")
plt.show()
