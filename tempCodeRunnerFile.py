import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==============================================================================
# 設定項目
# ==============================================================================
# 前処理済みデータセットのパス
TRAIN_DATASET_PATH = 'datasets/train_dataset.npy'
VAL_DATASET_PATH = 'datasets/validation_dataset.npy'

# 学習済みモデルの保存先
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'autoencoder_model.keras')

# 学習のハイパーパラメータ
EPOCHS = 20  # データセットを何周学習させるか
BATCH_SIZE = 4 # 一度に何個のデータを見て学習するか (PCのメモリに応じて調整)

# ==============================================================================
# モデル構築
# ==============================================================================
def build_convlstm_autoencoder(input_shape):
    """ConvLSTM Autoencoderモデルを構築する関数"""
    # エンコーダ（入力データを圧縮していく部分）
    encoder = models.Sequential([
        layers.Input(shape=input_shape),
        layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu'),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, activation='relu'),
        layers.BatchNormalization()
    ], name="encoder")

    # デコーダ（圧縮されたデータから元に戻していく部分）
    decoder = models.Sequential([
        layers.Input(shape=encoder.output_shape[1:]),
        layers.Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu'),
        layers.UpSampling3D(size=(2, 2, 2)), # 3Dでアップサンプリング
        layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu'),
        layers.UpSampling3D(size=(2, 2, 2)),
        layers.Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same', activation='sigmoid')
    ], name="decoder")
    
    # 実際にはエンコーダとデコーダの接続が複雑なため、Functional APIでモデルを構築
    # この部分は少し複雑ですが、定型的な書き方として捉えてください。
    input_seq = layers.Input(shape=input_shape)
    encoded_seq = encoder(input_seq)
    
    # デコーダが3Dの入力を期待するので、時間軸を拡張
    encoded_seq_expanded = layers.Reshape(target_shape=(1,)+encoder.output_shape[1:])(encoded_seq)
    # デコーダに合うようにリピート
    repeated_vector = layers.RepeatVector(input_shape[0])(encoded_seq)

    # 再構築したデコーダ部分
    x = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(repeated_vector)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    decoded_seq = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(x)
    
    autoencoder = models.Model(input_seq, decoded_seq)
    return autoencoder

# ==============================================================================
# メイン処理
# ==============================================================================
if __name__ == '__main__':
    # --- 1. データの読み込み ---
    print("Loading datasets...")
    x_train = np.load(TRAIN_DATASET_PATH)
    x_val = np.load(VAL_DATASET_PATH)
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    
    # --- 2. モデルの構築 ---
    # 入力データの形状を取得 (サンプル数を除く)
    input_shape = x_train.shape[1:]
    model = build_convlstm_autoencoder(input_shape)
    
    # モデルのコンパイル（学習方法の設定）
    model.compile(optimizer='adam', loss='mse') # mse: 平均二乗誤差
    model.summary()

    # --- 3. モデルの学習 ---
    print("\nStarting training...")
    history = model.fit(
        x_train,
        x_train, # オートエンコーダなので、入力と出力が同じ
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, x_val), # 検証データも入力と出力が同じ
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
        ]
    )
    print("Training finished.")

    # --- 4. 学習済みモデルの保存 ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # --- 5. 学習曲線のプロットと保存 ---
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('learning_curve.png')
    print("Learning curve saved to learning_curve.png")