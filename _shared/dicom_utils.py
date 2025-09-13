# _shared/dicom_utils.py
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image

def dicom_to_pil(dcm_path: str, to_uint8: bool = True) -> Image.Image:
    """
    RSNA CXR DICOM을 안전하게 읽어 PIL(gray)로 변환.
    - VOI LUT(window) 우선 적용
    - PhotometricInterpretation == MONOCHROME1이면 반전
    - 12~16bit를 8bit로 안전 스케일
    """
    ds = pydicom.dcmread(dcm_path, force=True)
    arr = ds.pixel_array

    # VOI LUT / Window
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    arr = arr.astype(np.float32)

    # PhotometricInterpretation
    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        # MONOCHROME1: 값이 클수록 어두움 → 반전
        arr = arr.max() - arr

    # 16bit → 8bit 스케일링
    if to_uint8:
        vmin, vmax = np.percentile(arr, (0.5, 99.5))
        if vmax <= vmin:
            vmin, vmax = arr.min(), arr.max()
        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
        arr = (arr * 255.0).astype(np.uint8)
    else:
        # 0~1 float
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255.0).astype(np.uint8)

    return Image.fromarray(arr, mode="L")
