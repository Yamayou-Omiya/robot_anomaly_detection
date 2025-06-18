import os
import cv2
import numpy as np
from tqdm import tqdm # 進捗バーを表示するためにtqdmをインポート

# --- 設定項目 ---
# これらの値は、PCのスペックや動画の内容に応じて調整します。
IMG_SIZE = 128         # 画像のサイズ（高さと幅）
SEQUENCE_LENGTH = 30   # 1つのデータとする動画のフレーム数（シーケンス長）

# 入力と出力のパス
TRAIN_DATA_DIR = './data/train'
VAL_DATA_DIR = './data/validation'
OUTPUT_TRAIN_FILE = 'train_dataset.npy'
OUTPUT_VAL_FILE = 'validation_dataset.npy'


def create_dataset(data_dir, output_file, img_size, seq_length):
    """
    指定されたディレクトリ内の動画を前処理し、npyファイルとして保存する関数
    """
    print(f"Processing videos from: {data_dir}")
    
    # ディレクトリ内の動画ファイルパスを取得
    video_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.mp4', '.avi', '.mov'))])
    
    all_sequences = []

    # tqdmを使って進捗を可視化
    for video_path in tqdm(video_files):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < seq_length:
            ret, frame = cap.read()
            if not ret:
                # 動画のフレームが尽きたらループを抜ける
                break
            
            # --- ここでフレームを加工 ---
            # 1. グレースケールに変換
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 2. リサイズ
            resized_frame = cv2.resize(gray_frame, (img_size, img_size))
            # 3. 0.0～1.0の範囲に正規化
            normalized_frame = resized_frame / 255.0
            
            frames.append(normalized_frame)
        
        cap.release()
        
        # フレーム数が指定したシーケンス長に達した場合のみデータセットに追加
        if len(frames) == seq_length:
            all_sequences.append(frames)

    # PythonのリストをNumPy配列に変換
    dataset = np.array(all_sequences, dtype=np.float32)

    # 4次元配列（バッチ、時間、高さ、幅）に拡張
    # CNN（Conv2D）は通常4次元の入力を期待するため、チャンネル次元を追加します
    dataset = np.expand_dims(dataset, axis=-1)
    
    # ファイルに保存
    np.save(output_file, dataset)
    
    print(f"Dataset saved to {output_file}")
    print(f"Shape of the dataset: {dataset.shape}")
    # 期待されるShape: (動画の本数, シーケンス長, 高さ, 幅, チャンネル数=1)
    # 例: (80, 30, 128, 128, 1)


# --- スクリプトの実行 ---
if __name__ == '__main__':
    # 訓練データの作成
    create_dataset(TRAIN_DATA_DIR, OUTPUT_TRAIN_FILE, IMG_SIZE, SEQUENCE_LENGTH)
    
    # 検証データの作成
    create_dataset(VAL_DATA_DIR, OUTPUT_VAL_FILE, IMG_SIZE, SEQUENCE_LENGTH)

    print("All preprocessing finished!")