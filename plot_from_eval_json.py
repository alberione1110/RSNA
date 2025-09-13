import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "_reports"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 평가 JSON이 들어있는 실험 폴더들
EXPS = [
    ("basic",                "eval_result.json"),
    ("pixelShuffle-only",    "eval_result.json"),
    ("partShuffle-only",     "eval_result.json"),
    ("chaotic-only",         "eval_result.json"),
    ("chaotic_pixelShuffle", "eval_result.json"),
    ("chaotic_partShuffle",  "eval_result.json"),
]

def load_jsons():
    rows = []
    found_any = False
    for folder, jf in EXPS:
        jpath = ROOT / folder / jf
        if not jpath.exists():
            print(f"⚠️  Skip (json not found): {jpath}")
            continue
        with open(jpath, "r") as f:
            data = json.load(f)
        m = data.get("metrics", {})
        rows.append({
            "folder": folder,
            "mode": data.get("mode", folder),
            "acc": m.get("accuracy", np.nan),
            "precision": m.get("precision", np.nan),
            "recall": m.get("recall", np.nan),
            "f1": m.get("f1", np.nan),
            "roc_auc": m.get("roc_auc", np.nan),
            "ap": m.get("ap", np.nan),
            "tn": m.get("tn", 0),
            "fp": m.get("fp", 0),
            "fn": m.get("fn", 0),
            "tp": m.get("tp", 0),
        })
        found_any = True
    if not rows:
        raise FileNotFoundError("평가 JSON을 찾지 못했습니다. 각 실험 폴더에서 evaluate를 먼저 실행하세요.")
    df = pd.DataFrame(rows)
    df = df.sort_values("folder")
    csv_path = OUTDIR / "summary_from_json.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved summary CSV -> {csv_path}")
    return df

def plot_bars(df):
    plt.figure(figsize=(10,5))
    x = np.arange(len(df))
    width = 0.25
    plt.bar(x - width, df["acc"], width, label="Acc")
    plt.bar(x,         df["f1"],  width, label="F1")
    plt.bar(x + width, df["roc_auc"], width, label="ROC-AUC")
    plt.xticks(x, df["folder"], rotation=20)
    plt.ylim(0, 1.0)
    plt.title("Summary (from JSON): Acc / F1 / ROC-AUC")
    plt.legend()
    plt.tight_layout()
    out = OUTDIR / "json_summary_bars.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✅ Saved {out}")

def plot_confusion_grids(df):
    """
    각 실험의 혼동행렬을 개별 이미지로 저장.
    JSON에는 tn,fp,fn,tp가 있으니 바로 그려줌.
    """
    for _, row in df.iterrows():
        folder = row["folder"]
        tn, fp, fn, tp = int(row["tn"]), int(row["fp"]), int(row["fn"]), int(row["tp"])
        cm = np.array([[tn, fp],
                       [fn, tp]])

        fig = plt.figure(figsize=(3.5,3.5))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=12)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["True 0","True 1"])
        ax.set_title(f"Confusion Matrix: {folder}")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        out = OUTDIR / f"cm_{folder}.png"
        plt.savefig(out, dpi=200)
        plt.close(fig)
        print(f"✅ Saved {out}")

def plot_curves_if_npz():
    """
    JSON만으로는 ROC/PR 곡선이 불가(확률 필요).
    하지만 각 폴더에 eval_preds.npz가 있으면 곡선도 한번에 그려준다.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

    # ROC
    fig1 = plt.figure(figsize=(7,6))
    ax1 = fig1.add_subplot(111)
    any_curve = False

    # PR
    fig2 = plt.figure(figsize=(7,6))
    ax2 = fig2.add_subplot(111)

    for folder, _ in EXPS:
        npz_path = ROOT / folder / "eval_preds.npz"
        if not npz_path.exists():
            print(f"ℹ️  No preds for curves: {npz_path}")
            continue
        npz = np.load(npz_path)
        y_true = npz["y_true"]
        y_prob = npz["y_prob"]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        pr, rc, _ = precision_recall_curve(y_true, y_prob)
        ax1.plot(fpr, tpr, label=f"{folder} (AUC={auc(fpr,tpr):.3f})")
        ax2.plot(rc, pr,  label=f"{folder} (AP={average_precision_score(y_true,y_prob):.3f})")
        any_curve = True

    if any_curve:
        ax1.plot([0,1],[0,1],"--",color="gray",linewidth=1)
        ax1.set_title("ROC Curves (from preds.npz)")
        ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout(); 
        out1 = OUTDIR / "json_roc_curves.png"
        fig1.savefig(out1, dpi=200); plt.close(fig1)
        print(f"✅ Saved {out1}")

        ax2.set_title("Precision-Recall Curves (from preds.npz)")
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout(); 
        out2 = OUTDIR / "json_pr_curves.png"
        fig2.savefig(out2, dpi=200); plt.close(fig2)
        print(f"✅ Saved {out2}")
    else:
        plt.close(fig1); plt.close(fig2)
        print("ℹ️  eval_preds.npz를 찾지 못해 ROC/PR 곡선은 건너뜁니다.")

if __name__ == "__main__":
    df = load_jsons()
    plot_bars(df)
    plot_confusion_grids(df)
    plot_curves_if_npz()
    print(f"\n📁 All outputs saved in: {OUTDIR}")
