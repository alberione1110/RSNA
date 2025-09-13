import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "_shared"))

from encrypt_ops import (
    apply_identity, apply_shuffle_only, apply_partshuffle_only,
    apply_xor_only, apply_xor_then_shuffle, apply_xor_then_partshuffle
)
from dataset import RSNADataset
from train_util import default_transform, train_loop
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "rsna-pneumonia-detection-challenge/stage_2_train_images"
CSV_PATH = "rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
IMAGE_SIZE, BATCH, EPOCHS, LR = 224, 32, 30, 1e-4
CHAOTIC_KEY = "my_secret_key"
BLOCK_SIZE = 8

def load_label_map(csv_path):
    df = pd.read_csv(csv_path)
    return {str(r["patientId"]): int(r["Target"]) for _, r in df.iterrows()}

def run_one(mode, preprocess_fn, kwargs, save_name):
    print(f"\n=== Running {mode} ===")
    label_map = load_label_map(CSV_PATH)
    ids = list(label_map.keys())
    tr_ids, va_ids = train_test_split(ids, test_size=0.2, random_state=42)

    tfm = default_transform(IMAGE_SIZE)

    tr_ds = RSNADataset(DATA_DIR, label_map, transform=tfm,
                        filter_ids=tr_ids, preprocess_fn=preprocess_fn,
                        preprocess_kwargs=kwargs)
    va_ds = RSNADataset(DATA_DIR, label_map, transform=tfm,
                        filter_ids=va_ids, preprocess_fn=preprocess_fn,
                        preprocess_kwargs=kwargs)

    model_path = f"best_model_{save_name}.pth"
    train_loop(tr_ds, va_ds, batch_size=BATCH, lr=LR,
               num_epochs=EPOCHS, model_save_path=model_path)

if __name__ == "__main__":
    run_one("basic", apply_identity, {}, "basic")
    run_one("pixelShuffle", apply_shuffle_only, {}, "pixelShuffle")
    run_one("partShuffle", apply_partshuffle_only, {"block": BLOCK_SIZE}, "partShuffle")
    run_one("chaotic-only", apply_xor_only, {"key": CHAOTIC_KEY}, "chaoticOnly")
    run_one("chaotic_pixelShuffle", apply_xor_then_shuffle, {"key": CHAOTIC_KEY}, "chaotic_pixelShuffle")
    run_one("chaotic_partShuffle", apply_xor_then_partshuffle,
            {"key": CHAOTIC_KEY, "block": BLOCK_SIZE}, "chaotic_partShuffle")
