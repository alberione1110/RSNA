# chaotic-only/train.py
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "_shared"))
from encrypt_ops import apply_xor_only
from dataset import RSNADataset
from train_util import default_transform, train_loop
from sklearn.model_selection import train_test_split
import pandas as pd

DATA_DIR = "../rsna-pneumonia-detection-challenge/train_png"
CSV_PATH = "../rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
MODEL_OUT = "best_model_chaoticOnly.pth"
CHAOTIC_KEY = "my_secret_key"
IMAGE_SIZE = 224
BATCH, EPOCHS, LR = 32, 30, 1e-4

def load_label_map(csv_path):
    df = pd.read_csv(csv_path)
    return {str(r["patientId"]): int(r["Target"]) for _, r in df.iterrows()}

if __name__ == "__main__":
    label_map = load_label_map(CSV_PATH)
    ids = list(label_map.keys())
    tr_ids, va_ids = train_test_split(ids, test_size=0.2, random_state=42)

    tfm = default_transform(IMAGE_SIZE)

    tr_ds = RSNADataset(DATA_DIR, label_map, transform=tfm, filter_ids=tr_ids,
                        preprocess_fn=apply_xor_only,
                        preprocess_kwargs={"key": CHAOTIC_KEY})
    va_ds = RSNADataset(DATA_DIR, label_map, transform=tfm, filter_ids=va_ids,
                        preprocess_fn=apply_xor_only,
                        preprocess_kwargs={"key": CHAOTIC_KEY})

    train_loop(tr_ds, va_ds, batch_size=BATCH, lr=LR, num_epochs=EPOCHS,
               model_save_path=MODEL_OUT)
