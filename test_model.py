import os
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm

# --- 設定項目 ---
MODEL_PATH = 'models/future_prediction_model.keras'
THRESHOLD_FILE = 'prediction_threshold.txt'
IMG_SIZE = 48
SEQUENCE_LENGTH = 60
NORMAL_DIR = 'data/test_normal'
ANOMALY_DIR = 'data/test_anomaly'


def preprocess_single_video(video_path, img_size, seq_length):
    """最後のseq_lengthフレームを切り出して前処理する"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    
    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = total_frames - seq_length
        if start_frame < 0: start_frame = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while len(frames) < seq_length:
            ret, frame = cap.read()
            if not ret: return None
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (img_size, img_size))
            normalized_frame = resized_frame / 255.0
            frames.append(normalized_frame)
    finally:
        cap.release()

    if len(frames) != seq_length: return None
    return np.array(frames, dtype=np.float32)

def evaluate_folder(folder_path, model, threshold, expected_as_normal):
    """フォルダ内の動画を評価する"""
    print(f"\n--- Testing videos in: {folder_path} ---")
    video_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.MOV'))])
    if not video_files:
        print("No videos found.")
        return 0, 0

    correct_predictions, total_videos = 0, len(video_files)
    for video_path in tqdm(video_files, desc="Evaluating"):
        video_sequence = preprocess_single_video(video_path, IMG_SIZE, SEQUENCE_LENGTH)
        if video_sequence is None:
            print(f"[ERROR] Could not process: {video_path}")
            continue
            
        # 入力と正解に分割
        input_seq = np.expand_dims(video_sequence[:-1], axis=0)
        target_frame = np.expand_dims(video_sequence[-1], axis=0)

        predicted_frame = model.predict(input_seq, verbose=0)
        error = np.mean(np.square(target_frame - predicted_frame))
        is_anomaly = error > threshold
        
        result_str = "ANOMALY" if is_anomaly else "Normal"
        print(f"[{result_str}] {os.path.basename(video_path)} (Error: {error:.6f})")

        if (expected_as_normal and not is_anomaly) or (not expected_as_normal and is_anomaly):
            correct_predictions += 1
            
    return correct_predictions, total_videos

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("Loading model and threshold...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(THRESHOLD_FILE, 'r') as f:
        threshold = float(f.read())
    print(f"Using Threshold: {threshold:.6f}")

    normal_correct, normal_total = evaluate_folder(NORMAL_DIR, model, threshold, expected_as_normal=True)
    anomaly_correct, anomaly_total = evaluate_folder(ANOMALY_DIR, model, threshold, expected_as_normal=False)

    print("\n" + "="*50 + "\n--- Final Test Summary ---\n" + "="*50)
    if normal_total > 0:
        print(f"Normal Videos Accuracy:  {normal_correct} / {normal_total} ({(normal_correct / normal_total) * 100:.1f}%)")
    if anomaly_total > 0:
        print(f"Anomaly Videos Accuracy: {anomaly_correct} / {anomaly_total} ({(anomaly_correct / anomaly_total) * 100:.1f}%)")
    print("="*50)