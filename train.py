import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- 設定項目 ---
TRAIN_DATASET_PATH = 'datasets/train_dataset.npy'
VAL_DATASET_PATH = 'datasets/validation_dataset.npy'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'future_prediction_model.keras') # モデル名を変更
EPOCHS = 30 # 少し長めに学習させます
BATCH_SIZE = 4

# --- GPUメモリ成長を有効にする設定 ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def build_prediction_model(input_shape):
    """未来の1フレームを予測するモデルを構築する関数"""
    # input_shape: (シーケンス長-1, 高さ, 幅, チャンネル数) e.g. (29, 64, 64, 1)
    inputs = layers.Input(shape=input_shape)

    # エンコーダ：動画シーケンスから動きの特徴を圧縮
    x = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, activation='tanh')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=False, activation='tanh')(x)
    x = layers.BatchNormalization()(x)

    # デコーダ：圧縮された特徴から、未来の1フレームを生成
    # Conv2DTransposeを使って、画像の解像度を復元していく
    # (64, 64, 16) -> (1, 64, 64, 16) に次元を拡張
    x = layers.Reshape((input_shape[1], input_shape[2], 16))(x)
    
    x = layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), padding='same', activation='tanh')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same', activation='tanh')(x)
    x = layers.BatchNormalization()(x)
    # 最終的に1チャンネルの画像（予測フレーム）を出力
    outputs = layers.Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, outputs)
    return model

if __name__ == '__main__':
    # --- 1. データの読み込みと分割 ---
    print("Loading datasets...")
    x_train_full = np.load(TRAIN_DATASET_PATH)
    x_val_full = np.load(VAL_DATASET_PATH)

    # データセットを「入力(最初のn-1フレーム)」と「正解(最後の1フレーム)」に分割
    x_train_input = x_train_full[:, :-1, ...]
    x_train_target = x_train_full[:, -1, ...]
    x_val_input = x_val_full[:, :-1, ...]
    x_val_target = x_val_full[:, -1, ...]

    print(f"Training input shape: {x_train_input.shape}")
    print(f"Training target shape: {x_train_target.shape}")

    # --- 2. モデルの構築 ---
    input_shape = x_train_input.shape[1:]
    model = build_prediction_model(input_shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()

    # --- 3. モデルの学習 ---
    print("\nStarting training...")
    history = model.fit(
        x_train_input,
        x_train_target, # 正解データとして最後のフレームを与える
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val_input, x_val_target),
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
    plt.title('Learning Curve (Future Prediction Model)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('prediction_learning_curve.png') # グラフのファイル名を変更
    print("Learning curve saved to prediction_learning_curve.png")