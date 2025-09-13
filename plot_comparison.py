"""
RSNA/export_transforms_and_plots.py

기능:
1) 검증셋에서 샘플 N개를 뽑아 6가지 케이스(원본 포함)로 변환한 PNG 저장
2) 샘플별 2x3 비교 그리드 저장 (모든 케이스)
3) 각 실험 폴더의 eval_result.json을 읽어 2x3 혼동행렬 그리드 저장

출력:
- RSNA/_reports/samples/<patientId>_<mode>.png  (개별 변환 이미지)
- RSNA/_reports/sample_grids/<patientId>_grid.png  (2x3 비교 그리드)
- RSNA/_reports/confusion_grid.png  (2x3 혼동행렬 비교)
"""

import os, sys, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

# 내부 모듈 경로
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "_shared"))

from dicom_utils import dicom_to_pil
from encrypt_ops import (
    apply_identity,
    apply_shuffle_only,
    apply_partshuffle_only,
    apply_xor_only,
    apply_xor_then_shuffle,
    apply_xor_then_partshuffle,
)
# NOTE: 여기서는 PIL 이미지를 직접 변환/저장하므로 dataset/transform은 안 써도 됨.

# ====== 경로/환경 설정 ======
DATA_DIR = ROOT / "rsna-pneumonia-detection-challenge" / "stage_2_train_images"
CSV_PATH = ROOT / "rsna-pneumonia-detection-challenge" / "stage_2_train_labels.csv"

OUTDIR = ROOT / "_reports"
SAMPLES_DIR = OUTDIR / "samples"
GRIDS_DIR = OUTDIR / "sample_grids"
OUTDIR.mkdir(exist_ok=True, parents=True)
SAMPLES_DIR.mkdir(exist_ok=True, parents=True)
GRIDS_DIR.mkdir(exist_ok=True, parents=True)

# Grad/암호화 설정
CHAOTIC_KEY = "my_secret_key"
BLOCK_SIZE = 8

# 몇 개의 샘플을 그릴지 (검증셋에서 임의 선택)
NUM_SAMPLES = 4
VAL_SEED = 42

# 케이스 목록(이 순서로 2x3 그리드)
CASES = [
    ("Original",                   apply_identity,              {}),
    ("Pixel Random",               apply_shuffle_only,          {}),
    ("Part Shuffle",               apply_partshuffle_only,      {"block": BLOCK_SIZE}),
    ("Chaotic Only",               apply_xor_only,              {"key": CHAOTIC_KEY}),
    ("Chaotic + Pixel Random",     apply_xor_then_shuffle,      {"key": CHAOTIC_KEY}),
    ("Chaotic + Part Shuffle",     apply_xor_then_partshuffle,  {"key": CHAOTIC_KEY, "block": BLOCK_SIZE}),
]

# eval_result.json 위치 (혼동행렬 그리드용)
EVAL_JSONS = {
    "Original":                 ROOT / "basic"                / "eval_result.json",
    "Pixel Random":             ROOT / "pixelShuffle-only"    / "eval_result.json",
    "Part Shuffle":             ROOT / "partShuffle-only"     / "eval_result.json",
    "Chaotic Only":             ROOT / "chaotic-only"         / "eval_result.json",
    "Chaotic + Pixel Random":   ROOT / "chaotic_pixelShuffle" / "eval_result.json",
    "Chaotic + Part Shuffle":   ROOT / "chaotic_partShuffle"  / "eval_result.json",
}

# --------------------------
# 유틸
# --------------------------
def load_label_map(csv_path: Path):
    df = pd.read_csv(csv_path)
    return {str(r["patientId"]): int(r["Target"]) for _, r in df.iterrows()}

def load_one_dicom(patient_id: str):
    dcm_path = DATA_DIR / f"{patient_id}.dcm"
    if not dcm_path.exists():
        raise FileNotFoundError(f"DICOM not found: {dcm_path}")
    return dicom_to_pil(str(dcm_path))  # PIL "L"

def pil_to_gray_np(pil_img):
    """PIL L → np.uint8 2D (H,W)."""
    arr = np.array(pil_img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr

def save_gray_png(np_gray, path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(path), np_gray)  # grayscale 저장

def make_grid_2x3(images_dict, out_path: Path, suptitle=None):
    """
    images_dict: {title: np.ndarray(gray)}
    """
    titles = [c[0] for c in CASES]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes = axes.flatten()
    for i, title in enumerate(titles):
        img = images_dict.get(title)
        ax = axes[i]
        if img is None:
            ax.axis("off")
            continue
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, y=0.98, fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def export_transforms_for_patient(patient_id: str):
    """
    단일 patientId에 대해 6케이스 PNG 저장 + 2x3 그리드 저장
    """
    original = load_one_dicom(patient_id)

    case_images = {}
    for title, fn, kwargs in CASES:
        pil_out = fn(original, img_id=patient_id, **kwargs)
        np_gray = pil_to_gray_np(pil_out)
        case_images[title] = np_gray
        save_gray_png(np_gray, SAMPLES_DIR / f"{patient_id}_{title.replace(' ','_')}.png")

    make_grid_2x3(case_images, GRIDS_DIR / f"{patient_id}_grid.png", suptitle=f"ID: {patient_id}")

def export_random_samples(num_samples=4, seed=42):
    label_map = load_label_map(CSV_PATH)
    ids = sorted(list(label_map.keys()))
    _, val_ids = train_test_split(ids, test_size=0.2, random_state=seed)
    picks = random.sample(val_ids, k=min(num_samples, len(val_ids)))
    for pid in picks:
        print(f"→ Exporting transforms for {pid}")
        export_transforms_for_patient(pid)

def confusion_grid_from_json(out_path: Path):
    """
    6케이스의 eval_result.json에서 혼동행렬을 읽어 2x3 heatmap 그리드로 저장.
    값은 비율(0~1)로 표기.
    """
    titles = [c[0] for c in CASES]
    cms = {}
    for title in titles:
        jpath = EVAL_JSONS.get(title)
        if jpath is None or not jpath.exists():
            print(f"⚠️  Skip (json not found): {jpath}")
            continue
        with open(jpath, "r") as f:
            data = json.load(f)
        tn = int(data["metrics"]["tn"])
        fp = int(data["metrics"]["fp"])
        fn = int(data["metrics"]["fn"])
        tp = int(data["metrics"]["tp"])
        total = max(1, tn + fp + fn + tp)
        cm = np.array([[tn, fp], [fn, tp]], dtype=np.float32) / float(total)
        cms[title] = cm

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes = axes.flatten()
    im = None
    for i, title in enumerate(titles):
        ax = axes[i]
        cm = cms.get(title)
        if cm is None:
            ax.axis("off")
            continue
        im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", color="white", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Neg", "Pred: Pos"], fontsize=9)
        ax.set_yticklabels(["True: Neg", "True: Pos"], fontsize=9)
    # 공통 컬러바
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
        cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved {out_path}")

# --------------------------
# 메인
# --------------------------
if __name__ == "__main__":
    # 1) 샘플 N개 자동 추출 & 모든 케이스 변환 이미지 저장 + 2x3 그리드 생성
    export_random_samples(num_samples=NUM_SAMPLES, seed=VAL_SEED)

    # 2) 혼동행렬 2x3 그리드 생성 (6케이스)
    confusion_grid_from_json(OUTDIR / "confusion_grid.png")

    print(f"\n📁 Outputs:")
    print(f"- Transformed PNGs: {SAMPLES_DIR}")
    print(f"- Sample grids:     {GRIDS_DIR}")
    print(f"- Confusion grid:   {OUTDIR/'confusion_grid.png'}")
