"""Excel/CSV dışa aktarma ve oturum kaydet/yükle."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime

import pandas as pd


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def sheets_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Birden çok DataFrame'i tek Excel dosyasına (her biri ayrı sayfa) yazar."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                safe = str(name)[:31].replace("/", "-").replace("\\", "-")
                df.to_excel(writer, sheet_name=safe or "Sayfa", index=False)
    buf.seek(0)
    return buf.getvalue()


def save_session(df: pd.DataFrame, meta: dict) -> bytes:
    """Veri + ayarları .statlab (zip) dosyasına paketler."""
    buf = io.BytesIO()
    meta = dict(meta)
    meta["saved_at"] = datetime.now().isoformat()
    meta["format"] = "statlab-v1"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.csv", df.to_csv(index=False))
        z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    buf.seek(0)
    return buf.getvalue()


def load_session(data: bytes) -> tuple[pd.DataFrame, dict]:
    """.statlab dosyasını geri yükler."""
    buf = io.BytesIO(data)
    with zipfile.ZipFile(buf, "r") as z:
        with z.open("data.csv") as f:
            df = pd.read_csv(f)
        meta = {}
        if "meta.json" in z.namelist():
            with z.open("meta.json") as f:
                meta = json.loads(f.read().decode("utf-8"))
    return df, meta
