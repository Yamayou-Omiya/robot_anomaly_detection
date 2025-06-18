import os
import cv2
import numpy as np
from tqdm import tqdm

# ==============================================================================
# 設定項目
# ==============================================================================
# 画像のサイズ（高さと幅）
# 128x128ピクセルにリサイズします。
IMG_SIZE = 128

# 1つのデータとする動画のフレーム数（シーケンス長）
# 1つの動画から30フレームを抜き出して1つのデータセットとします。
SEQUENCE_LENGTH = 150

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
    指定されたディレクトリ内の動画を前処理し、npyファイルとして保存する関数

    Args:
        data_dir (str): 動画ファイルが格納されているディレクトリのパス
        output_file (str): 出力するnpyファイルのパス
        img_size (int): リサイズ後の画像のサイズ
        seq_length (int): 1データあたりのシーケンス長（フレーム数）
    """
    print(f"----- Processing videos from: {data_dir} -----")

    # ディレクトリ内の動画ファイルパスをリストとして取得
    video_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(('.mp4', '.avi', '.mov', '.MOV')) # .MOVなど大文字にも対応
    ])

    if not video_files:
        print(f"Warning: No video files found in {data_dir}. Skipping.")
        # 空の配列を表すnpyファイルを作成して終了
        np.save(output_file, np.array([]))
        print(f"Empty dataset saved to {output_file}")
        return

    all_sequences = []

    # tqdmを使って進捗バーを表示
    for video_path in tqdm(video_files, desc="Processing videos"):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < seq_length:
            ret, frame = cap.read()
            if not ret:
                break  # 動画のフレームが尽きたか、読み込みエラー

            # --- フレームの加工処理 ---
            # 1. グレースケール（白黒）に変換
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 2. 指定したサイズにリサイズ
            resized_frame = cv2.resize(gray_frame, (img_size, img_size))
            # 3. ピクセル値を0.0～1.0の範囲に正規化
            normalized_frame = resized_frame / 255.0

            frames.append(normalized_frame)

        cap.release()

        # フレーム数が指定したシーケンス長に達した場合のみデータセットに追加
        if len(frames) == seq_length:
            all_sequences.append(frames)

    # PythonのリストをNumPy配列に変換
    dataset = np.array(all_sequences, dtype=np.float32)
    
    # ConvLSTMモデルなどに入力するために、チャンネルの次元を追加
    # (バッチ, 時間, 高さ, 幅) -> (バッチ, 時間, 高さ, 幅, チャンネル=1)
    if dataset.ndim == 4:
        dataset = np.expand_dims(dataset, axis=-1)

    # 最終的なデータセットをファイルに保存
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