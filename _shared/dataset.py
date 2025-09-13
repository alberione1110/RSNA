# _shared/dataset.py
import os, torch, glob
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Dict, Optional
from dicom_utils import dicom_to_pil

class RSNADataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_map: Dict[str, int],
        transform: Optional[Callable] = None,
        filter_ids=None,
        preprocess_fn: Optional[Callable] = None,
        preprocess_kwargs: Optional[dict] = None,
    ):
        self.image_dir = image_dir
        self.label_map = label_map
        self.transform = transform
        self.preprocess_fn = preprocess_fn
        self.preprocess_kwargs = preprocess_kwargs or {}

        all_ids = list(label_map.keys())
        if filter_ids:
            keep = set(filter_ids)
            all_ids = [pid for pid in all_ids if pid in keep]
        self.image_ids = all_ids

    def _load_image_any(self, patient_id: str) -> Image.Image:
        # 1) DICOM 우선
        dcm = os.path.join(self.image_dir, f"{patient_id}.dcm")
        if os.path.exists(dcm):
            return dicom_to_pil(dcm)

        # 2) PNG/JPG 대체
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(self.image_dir, f"{patient_id}{ext}")
            if os.path.exists(p):
                return Image.open(p).convert("L")

        # 3) 혹시 파일명이 다르면 글롭으로 검색
        found = glob.glob(os.path.join(self.image_dir, f"*{patient_id}*"))
        for p in found:
            if p.lower().endswith(".dcm"):
                return dicom_to_pil(p)
            elif p.lower().endswith((".png", ".jpg", ".jpeg")):
                return Image.open(p).convert("L")

        raise FileNotFoundError(f"Image for id={patient_id} not found in {self.image_dir}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        pid = self.image_ids[idx]
        img = self._load_image_any(pid)  # DICOM->PIL(gray) or PNG->PIL(gray)

        if self.preprocess_fn:
            img = self.preprocess_fn(img, img_id=pid, **self.preprocess_kwargs)

        if self.transform:
            img = self.transform(img)

        label = self.label_map[pid]
        return img, torch.tensor(label, dtype=torch.long)
