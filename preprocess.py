import os
import cv2
import numpy as np
from tqdm import tqdm

# ==============================================================================
# 設定項目
# ==============================================================================
# 画像のサイズ（高さと幅）
# 128x128ピクセルにリサイズします。
IMG_SIZE = 48

# 1つのデータとする動画のフレーム数（シーケンス長）
# 1つの動画から30フレームを抜き出して1つのデータセットとします。
SEQUENCE_LENGTH = 60

# --- 入力/出力パス ---

# 生の動画データが置かれているフォルダ
TRAIN_DATA_DIR = './data/train'
VAL_DATA_DIR = './data/validation'

# 前処理済みのデータセット(.npy)を保存するフォルダ
OUTPUT_DIR = 'datasets'

# ==============================================================================
# メイン処理
# ==============================================================================

def create_dataset(data_dir, output_file, img_size, seq_length):
    """
    指定されたディレクトリ内の動画の最後の部分を前処理し、npyファイルとして保存する関数
    """
    print(f"----- Processing last {seq_length} frames from videos in: {data_dir} -----")
    
    video_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(('.mp4', '.avi', '.mov', '.MOV'))
    ])

    if not video_files:
        print(f"Warning: No video files found in {data_dir}. Skipping.")
        np.save(output_file, np.array([]))
        print(f"Empty dataset saved to {output_file}")
        return

    all_sequences = []

    for video_path in tqdm(video_files, desc="Processing videos"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file {video_path}. Skipping.")
            continue

        try:
            # --- ここからが変更点 ---
            # 1. 動画の総フレーム数を取得
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 2. 読み取り開始フレームを計算 (総フレーム数 - 欲しいフレーム数)
            start_frame = total_frames - seq_length
            
            # もし動画が短すぎて開始フレームがマイナスになる場合は、先頭から読み取る
            if start_frame < 0:
                start_frame = 0

            # 3. 動画の読み取り位置を、計算した開始フレームにセットする
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            # --- 変更点ここまで ---

            frames = []
            while len(frames) < seq_length:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # フレームの高さ（height）を取得
                height = frame.shape[0]
                # フレームの高さの半分から下だけを切り出す
                cropped_frame = frame[height // 2:, :]
                gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                resized_frame = cv2.resize(gray_frame, (img_size, img_size))
                normalized_frame = resized_frame / 255.0
                frames.append(normalized_frame)

        finally:
            cap.release()

        if len(frames) == seq_length:
            all_sequences.append(frames)
        else:
            print(f"Warning: Could not extract {seq_length} frames from {video_path}. Only found {len(frames)}. Skipping.")


    dataset = np.array(all_sequences, dtype=np.float32)
    
    if dataset.ndim == 4:
        dataset = np.expand_dims(dataset, axis=-1)

    np.save(output_file, dataset)
    
    print(f"Dataset saved to {output_file}")
    print(f"Shape of the dataset: {dataset.shape}")


# --- このスクリプトが直接実行された場合に以下の処理を行う ---
if __name__ == '__main__':
    # 出力先フォルダが存在しない場合は自動で作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 出力ファイルのフルパスを定義
    output_train_file = os.path.join(OUTPUT_DIR, 'train_dataset.npy')
    output_val_file = os.path.join(OUTPUT_DIR, 'validation_dataset.npy')

    # 訓練データセットの作成
    create_dataset(TRAIN_DATA_DIR, output_train_file, IMG_SIZE, SEQUENCE_LENGTH)

    # 検証データセットの作成
    create_dataset(VAL_DATA_DIR, output_val_file, IMG_SIZE, SEQUENCE_LENGTH)

    print("\nAll preprocessing finished!")