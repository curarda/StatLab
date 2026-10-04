"""Veri yükleme yardımcıları: CSV, Excel, JSON."""
from __future__ import annotations

import io
import json
import pandas as pd


def load_csv(file, **kwargs) -> pd.DataFrame:
    """CSV dosyası veya buffer yükle. Ayırıcıyı otomatik dene."""
    # Streamlit UploadedFile ya da yol olabilir
    raw = _read_bytes(file)
    text = _decode(raw)
    # Ayırıcı sezgisi
    sep = kwargs.pop("sep", None)
    if sep is None:
        first_line = text.splitlines()[0] if text.splitlines() else ""
        if first_line.count(";") > first_line.count(","):
            sep = ";"
        elif first_line.count("\t") > first_line.count(","):
            sep = "\t"
        else:
            sep = ","
    df = pd.read_csv(io.StringIO(text), sep=sep, **kwargs)
    return _clean(df)


def load_excel(file, sheet_name=0, **kwargs) -> pd.DataFrame:
    raw = _read_bytes(file)
    df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name, **kwargs)
    if isinstance(df, dict):  # birden fazla sayfa
        df = next(iter(df.values()))
    return _clean(df)


def excel_sheet_names(file) -> list[str]:
    raw = _read_bytes(file)
    xls = pd.ExcelFile(io.BytesIO(raw))
    return xls.sheet_names


def load_json(file, **kwargs) -> pd.DataFrame:
    raw = _read_bytes(file)
    text = _decode(raw)
    data = json.loads(text)
    # Yaygın şekiller: kayıt listesi, sütun sözlüğü, {"data": [...]}
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], (list, dict)):
            data = data["data"]
    df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame(data)
    return _clean(df)


def load_any(file, filename: str | None = None, **kwargs) -> pd.DataFrame:
    """Uzantıya göre uygun yükleyiciyi seç."""
    name = (filename or getattr(file, "name", "") or "").lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return load_excel(file, **kwargs)
    if name.endswith(".json"):
        return load_json(file, **kwargs)
    return load_csv(file, **kwargs)


# ---- iç yardımcılar ----

def _read_bytes(file) -> bytes:
    if isinstance(file, (bytes, bytearray)):
        return bytes(file)
    if isinstance(file, str):
        with open(file, "rb") as fh:
            return fh.read()
    # dosya benzeri
    pos = file.tell() if hasattr(file, "tell") else None
    data = file.read()
    if pos is not None and hasattr(file, "seek"):
        file.seek(pos)
    if isinstance(data, str):
        return data.encode("utf-8")
    return data


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1254"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # Sütun adlarını sadeleştir
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Tamamen boş satır/sütunları at
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    # Sayıya çevrilebilen object sütunları çevir (virgüllü ondalıkları da dene)
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= 0.8 * df[col].notna().sum() and df[col].notna().sum() > 0:
                df[col] = converted
            else:
                # virgüllü ondalık dene
                try:
                    alt = pd.to_numeric(
                        df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                        errors="coerce",
                    )
                    if alt.notna().sum() >= 0.8 * df[col].notna().sum() and df[col].notna().sum() > 0:
                        df[col] = alt
                except Exception:
                    pass
    return df.reset_index(drop=True)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
