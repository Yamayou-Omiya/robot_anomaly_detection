import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm

# --- 設定項目 ---
# これらの設定は、他のスクリプトと完全に一致させる必要があります
MODEL_PATH = 'models/autoencoder_model.keras'
THRESHOLD_FILE = 'threshold.txt'
IMG_SIZE = 48
SEQUENCE_LENGTH = 60

# --- テスト対象のフォルダ ---
# ご自身のフォルダ名に合わせて、必要であれば修正してください
NORMAL_DIR = 'data/test_normal'
ANOMALY_DIR = 'data/test_anomaly'


def preprocess_single_video(video_path, img_size, seq_length):
    """
    1本の動画ファイルを読み込み、モデルに入力できる形式に前処理する関数（最後の1秒版）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video file {video_path}")
        return None
    
    frames = []
    try:
        # 動画の総フレーム数を取得
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 読み取り開始フレームを計算 (総フレーム数 - 欲しいフレーム数)
        start_frame = total_frames - seq_length
        if start_frame < 0:
            start_frame = 0 # 動画が短い場合は先頭から

        # 動画の読み取り位置を、計算した開始フレームにセット
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while len(frames) < seq_length:
            ret, frame = cap.read()
            if not ret:
                # フレームが足りずに動画が終わってしまった場合
                print(f"Warning: Could not read enough frames from {video_path}")
                return None
            height = frame.shape[0]
            cropped_frame = frame[height // 2:, :]
            # preprocess.pyと全く同じ加工処理
            gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (img_size, img_size))
            normalized_frame = resized_frame / 255.0
            frames.append(normalized_frame)
    finally:
        cap.release()
    
    # フレーム数が足りない場合は無効なデータとして扱う
    if len(frames) != seq_length:
        return None

    # モデルに入力するために、次元を整形する
    sequence = np.array(frames, dtype=np.float32)
    sequence = np.expand_dims(sequence, axis=-1)
    sequence = np.expand_dims(sequence, axis=0)
    return sequence

def evaluate_folder(folder_path, model, threshold, expected_as_normal):
    """指定されたフォルダ内の全動画を評価し、結果を返す関数"""
    print("\n" + "="*50)
    print(f"--- Testing videos in: {folder_path} ---")
    print("="*50)
    
    video_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(('.mp4', '.avi', '.mov', '.MOV'))
    ])
    
    if not video_files:
        print("No videos found in this directory.")
        return 0, 0

    correct_predictions = 0
    total_videos = len(video_files)

    for video_path in tqdm(video_files, desc="Evaluating"):
        video_sequence = preprocess_single_video(video_path, IMG_SIZE, SEQUENCE_LENGTH)
        if video_sequence is None:
            print(f"[ERROR] Could not process: {video_path}")
            continue

        reconstructed_sequence = model.predict(video_sequence, verbose=0)
        error = np.mean(np.square(video_sequence - reconstructed_sequence))
        
        is_anomaly = error > threshold
        
        # 判定結果の表示
        if is_anomaly:
            print(f"[ANOMALY] {video_path} (Error: {error:.6f})")
        else:
            print(f"[Normal]  {video_path} (Error: {error:.6f})")

        # 正解かどうかのカウント
        # 期待値が正常(True)で、異常と判定されなかった(False)場合 -> 正解
        if expected_as_normal and not is_anomaly:
            correct_predictions += 1
        # 期待値が異常(False)で、異常と判定された(True)場合 -> 正解
        elif not expected_as_normal and is_anomaly:
            correct_predictions += 1
            
    return correct_predictions, total_videos


# --- メイン処理 ---
if __name__ == '__main__':
    # GPUを無効化
    os.environ['CUDA_VISIBLE_DEDVICES'] = '-1'

    # 1. モデルと閾値の読み込み
    print("Loading model and threshold...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(THRESHOLD_FILE, 'r') as f:
            threshold = float(f.read())
    except IOError as e:
        print(f"Error: Could not load files. Please run train.py and evaluate.py first.")
        exit()

    print(f"Using Threshold: {threshold:.6f}")

    # 2. 正常動画フォルダの評価
    normal_correct, normal_total = evaluate_folder(NORMAL_DIR, model, threshold, expected_as_normal=True)
    
    # 3. 異常動画フォルダの評価
    anomaly_correct, anomaly_total = evaluate_folder(ANOMALY_DIR, model, threshold, expected_as_normal=False)

    # 4. 最終結果のサマリー
    print("\n" + "="*50)
    print("--- Final Test Summary ---")
    print("="*50)
    if normal_total > 0:
        normal_accuracy = (normal_correct / normal_total) * 100
        print(f"Normal Videos Accuracy:  {normal_correct} / {normal_total} ({normal_accuracy:.1f}%)")
    if anomaly_total > 0:
        anomaly_accuracy = (anomaly_correct / anomaly_total) * 100
        print(f"Anomaly Videos Accuracy: {anomaly_correct} / {anomaly_total} ({anomaly_accuracy:.1f}%)")
    print("="*50)