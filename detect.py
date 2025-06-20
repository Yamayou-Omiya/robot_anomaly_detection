import os
import sys
import numpy as np
import tensorflow as tf
import cv2

# --- 設定項目 ---
# これらの設定は、preprocess.pyやtrain.pyと完全に一致させる必要があります
MODEL_PATH = 'models/autoencoder_model.keras'
THRESHOLD_FILE = 'threshold.txt'
IMG_SIZE = 64
SEQUENCE_LENGTH = 30


def preprocess_single_video(video_path, img_size, seq_length):
    """1本の動画ファイルを読み込み、モデルに入力できる形式に前処理する関数"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while len(frames) < seq_length:
            ret, frame = cap.read()
            if not ret:
                # フレームが足りずに動画が終わってしまった場合
                return None
            
            # preprocess.pyと全く同じ加工処理
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (img_size, img_size))
            normalized_frame = resized_frame / 255.0
            frames.append(normalized_frame)
    finally:
        cap.release()
    
    # モデルに入力するために、次元を整形する
    sequence = np.array(frames, dtype=np.float32)
    sequence = np.expand_dims(sequence, axis=-1) # チャンネル次元 (白黒なので1)
    sequence = np.expand_dims(sequence, axis=0)  # バッチ次元 (1本の動画だけなので1)
    return sequence


# --- メイン処理 ---
if __name__ == '__main__':
    # GPUを無効化
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # コマンドラインから動画ファイルのパスを受け取る
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_video_file>")
        sys.exit(1)
    video_to_test = sys.argv[1]

    if not os.path.exists(video_to_test):
        print(f"Error: Video file not found at {video_to_test}")
        sys.exit(1)

    # 1. モデルと閾値の読み込み
    print("Loading model and threshold...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(THRESHOLD_FILE, 'r') as f:
            threshold = float(f.read())
    except IOError as e:
        print(f"Error: Could not load model or threshold file. Please run train.py and evaluate.py first.")
        print(e)
        sys.exit(1)

    # 2. テストする動画を前処理
    print(f"Preprocessing video: {video_to_test}...")
    video_sequence = preprocess_single_video(video_to_test, IMG_SIZE, SEQUENCE_LENGTH)

    if video_sequence is None:
        print("Error: Could not process the video. It might be too short or corrupted.")
        sys.exit(1)

    # 3. モデルで復元し、誤差を計算
    print("Calculating reconstruction error...")
    reconstructed_sequence = model.predict(video_sequence)
    error = np.mean(np.square(video_sequence - reconstructed_sequence))

    # 4. 閾値と比較して判定
    print("\n" + "="*40)
    print(f"  Reconstruction Error: {error:.6f}")
    print(f"  Threshold:            {threshold:.6f}")
    print("-"*40)
    if error > threshold:
        print("  Result: ANOMALY DETECTED! (異常を検知しました)")
    else:
        print("  Result: Normal (正常です)")
    print("="*40)