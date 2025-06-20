import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- 設定項目 ---
# 学習済みモデルが保存されているパス
MODEL_PATH = 'models/autoencoder_model.keras'
# 検証用データセットのパス (閾値の計算に使用)
VAL_DATASET_PATH = 'datasets/validation_dataset.npy'
# 計算した閾値を保存するファイル名
THRESHOLD_FILE = 'threshold.txt'
# 誤差の分布を保存するグラフのファイル名
DISTRIBUTION_CHART_FILE = 'reconstruction_error_distribution.png'


# --- メイン処理 ---
if __name__ == '__main__':
    # GPUを使わないように設定（この処理はCPUでも十分速いため）
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # 1. モデルと検証データの読み込み
    print("Loading model and validation data...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        x_val = np.load(VAL_DATASET_PATH)
    except IOError as e:
        print(f"Error: Could not load files. Please make sure training was completed.")
        print(e)
        exit()

    if x_val.size == 0:
        print("Error: Validation dataset is empty. Please check preprocess.py and data folders.")
        exit()

    # 2. 検証データで復元を行い、誤差を計算
    print("Calculating reconstruction errors on validation data...")
    reconstructed_x = model.predict(x_val)
    # 各動画クリップの誤差を計算 (平均二乗誤差)
    # (入力データ - 復元データ) の差をピクセルごとに計算し、二乗して、全ピクセルの平均をとる
    errors = np.mean(np.square(x_val - reconstructed_x), axis=(1, 2, 3, 4))
    print(f"Calculated errors for {len(errors)} validation samples.")

    # 3. 誤差の分布をグラフ(ヒストグラム)にして保存
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.8)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Number of samples")
    plt.title("Reconstruction error distribution on normal validation data")
    plt.savefig(DISTRIBUTION_CHART_FILE)
    print(f"Reconstruction error distribution chart saved to {DISTRIBUTION_CHART_FILE}")

    # 4. 閾値の計算と保存
    # 正常データの誤差分布から、外れ値と判断する閾値を決定する
    # ここでは例として「誤差の平均値 + 標準偏差の3倍」を閾値とする（統計学的に約99.7%のデータがこの範囲に収まる）
    threshold = np.mean(errors) + 3 * np.std(errors)
    
    print("\n" + "="*40)
    print(f"  Calculated Threshold: {threshold:.6f}")
    print("="*40 + "\n")

    with open(THRESHOLD_FILE, 'w') as f:
        f.write(str(threshold))
    print(f"Threshold saved to {THRESHOLD_FILE}")