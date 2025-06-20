import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- 設定項目 ---
MODEL_PATH = 'models/future_prediction_model.keras'
VAL_DATASET_PATH = 'datasets/validation_dataset.npy'
THRESHOLD_FILE = 'prediction_threshold.txt'
DISTRIBUTION_CHART_FILE = 'prediction_error_distribution.png'

# --- メイン処理 ---
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # 1. モデルと検証データの読み込み
    print("Loading model and validation data...")
    model = tf.keras.models.load_model(MODEL_PATH)
    x_val_full = np.load(VAL_DATASET_PATH)

    # 検証データを入力と正解に分割
    x_val_input = x_val_full[:, :-1, ...]
    x_val_target = x_val_full[:, -1, ...]
    
    # 2. 検証データで予測を行い、誤差を計算
    print("Calculating prediction errors on validation data...")
    predicted_frames = model.predict(x_val_input)
    
    # 各動画クリップの予測誤差を計算 (平均二乗誤差)
    errors = np.mean(np.square(x_val_target - predicted_frames), axis=(1, 2, 3))

    # 3. 誤差の分布をプロットして保存
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.8)
    plt.xlabel("Prediction Error")
    plt.ylabel("Number of samples")
    plt.title("Prediction error distribution on normal validation data")
    plt.savefig(DISTRIBUTION_CHART_FILE)
    print(f"Prediction error distribution chart saved to {DISTRIBUTION_CHART_FILE}")

    # 4. 閾値の計算と保存
    threshold = np.mean(errors) + 3 * np.std(errors)
    print(f"\nCalculated Threshold: {threshold:.6f}")

    with open(THRESHOLD_FILE, 'w') as f:
        f.write(str(threshold))
    print(f"Threshold saved to {THRESHOLD_FILE}")