import os
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
EPOCHS = 20
BATCH_SIZE = 4

# ==============================================================================
# モデル構築
# ==============================================================================
def build_convlstm_autoencoder(input_shape):
    """
    ConvLSTM Autoencoderモデルを構築する関数 (最終完成版)
    input_shape: (シーケンス長, 高さ, 幅, チャンネル数) e.g. (150, 128, 128, 1)
    """
    inputs = layers.Input(shape=input_shape)

    # --- エンコーダ (圧縮) ---
    x = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=False, activation='relu')(x)

    # --- 潜在表現をシーケンスに戻す処理 ---
    latent_vector = layers.Flatten()(x)
    x = layers.RepeatVector(input_shape[0])(latent_vector)
    x = layers.Reshape((input_shape[0], input_shape[1], input_shape[2], 16))(x)

    # --- デコーダ (復元) ---
    x = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # === ここが今回の修正点です ===
    # 3次元のカーネルサイズ (3, 3, 3) を正しく指定します
    outputs = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(inputs, outputs)
    return autoencoder

# ==============================================================================
# メイン処理
# ==============================================================================
if __name__ == '__main__':
    # GPUを強制的に無効化する設定（環境問題の切り分けのため）
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # --- 1. データの読み込み ---
    print("Loading datasets...")
    if not os.path.exists(TRAIN_DATASET_PATH) or not os.path.exists(VAL_DATASET_PATH):
        print("Error: Dataset files not found. Please run preprocess.py first.")
        exit()
    x_train = np.load(TRAIN_DATASET_PATH)
    x_val = np.load(VAL_DATASET_PATH)
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    
    # --- 2. モデルの構築 ---
    input_shape = x_train.shape[1:]
    model = build_convlstm_autoencoder(input_shape)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # --- 3. モデルの学習 ---
    print("\nStarting training...")
    history = model.fit(
        x_train,
        x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, x_val),
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