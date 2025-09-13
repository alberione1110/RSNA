import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# _shared 모듈
sys.path.append(os.path.join(os.path.dirname(__file__), "_shared"))
from eval import evaluate_experiment

# ===== 경로/공통설정 =====
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT/"rsna-pneumonia-detection-challenge/stage_2_train_images"
CSV_PATH = ROOT/"rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
IMAGE_SIZE = 224
BATCH_SIZE = 32
VAL_SEED = 42
CHAOTIC_KEY = "my_secret_key"
BLOCK = 8

# 각 실험 폴더와 체크포인트/모드 매핑
EXPS = [
    ("basic",                  "best_model_basic.pth",                  "basic",                  {}),
    ("pixelShuffle-only",      "best_model_pixelShuffle.pth",           "pixelShuffle-only",      {}),
    ("partShuffle-only",       "best_model_partShuffle.pth",            "partShuffle-only",       {"block": BLOCK}),
    ("chaotic-only",           "best_model_chaoticOnly.pth",            "chaotic-only",           {"key": CHAOTIC_KEY}),
    ("chaotic_pixelShuffle",   "best_model_chaotic_pixelShuffle.pth",   "chaotic_pixelShuffle",   {"key": CHAOTIC_KEY}),
    ("chaotic_partShuffle",    "best_model_chaotic_partShuffle.pth",    "chaotic_partShuffle",    {"key": CHAOTIC_KEY, "block": BLOCK}),
]

OUTDIR = ROOT/"_reports"
OUTDIR.mkdir(exist_ok=True, parents=True)

def run_all():
    rows = []
    for folder, ckpt, mode, extra in EXPS:
        exp_dir = ROOT/folder
        ckpt_path = exp_dir/ckpt
        print(f"\n=== EVAL: {folder} ({mode}) ===")
        res = evaluate_experiment(
            data_dir=str(DATA_DIR),
            csv_path=str(CSV_PATH),
            checkpoint=str(ckpt_path),
            mode=mode,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            key=extra.get("key"),
            block=extra.get("block", 8),
            out_path=str(exp_dir/"eval_result.json"),
            val_seed=VAL_SEED,
        )
        # 요약행 축적
        m = res["metrics"]
        rows.append({
            "folder": folder, "mode": mode, "ckpt": str(ckpt_path),
            "acc": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "roc_auc": m["roc_auc"], "ap": m["ap"],
            "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
        })

    df = pd.DataFrame(rows)
    csv_path = OUTDIR/"summary_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved summary CSV -> {csv_path}")

    # ===== 막대그래프(ACC/F1/ROC-AUC) =====
    plt.figure(figsize=(10,5))
    x = np.arange(len(df))
    width = 0.25
    plt.bar(x - width, df["acc"], width, label="Acc")
    plt.bar(x,         df["f1"],  width, label="F1")
    plt.bar(x + width, df["roc_auc"], width, label="ROC-AUC")
    plt.xticks(x, df["folder"], rotation=20)
    plt.ylim(0,1.0)
    plt.title("Summary: Acc / F1 / ROC-AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR/"summary_bars.png", dpi=200)
    plt.close()

    # ===== ROC / PR 곡선 (여러 실험 한 그림) =====
    # eval_preds.npz에서 y_true, y_prob 로드
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

    fig1 = plt.figure(figsize=(7,6))
    fig2 = plt.figure(figsize=(7,6))
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)

    y_true_ref = None
    for folder, _, _, _ in EXPS:
        npz = np.load(ROOT/folder/"eval_preds.npz")
        y_true = npz["y_true"]
        y_prob = npz["y_prob"]
        if y_true_ref is None:
            y_true_ref = y_true
        else:
            # 같은 검증셋인지 sanity-check (길이까지만 확인)
            assert len(y_true_ref) == len(y_true), "Eval splits mismatch length"

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        pr, rc, _ = precision_recall_curve(y_true, y_prob)
        ax1.plot(fpr, tpr, label=f"{folder} (AUC={auc(fpr,tpr):.3f})")
        ax2.plot(rc, pr,  label=f"{folder} (AP={average_precision_score(y_true,y_prob):.3f})")

    ax1.plot([0,1],[0,1],"--",color="gray",linewidth=1)
    ax1.set_title("ROC Curves")
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout(); fig1.savefig(OUTDIR/"roc_curves.png", dpi=200); plt.close(fig1)

    ax2.set_title("Precision-Recall Curves")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout(); fig2.savefig(OUTDIR/"pr_curves.png", dpi=200); plt.close(fig2)

    print(f"✅ Saved plots to {OUTDIR}")

if __name__ == "__main__":
    run_all()
