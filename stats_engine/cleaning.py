"""Veri temizleme: eksik değer, aykırı değer, dönüşüm."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import interpret as I


# ---------------- Eksik değer ----------------

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        n_miss = int(df[c].isna().sum())
        rows.append({"Sütun": c, "Tip": str(df[c].dtype), "Eksik": n_miss,
                     "Eksik %": round(df[c].isna().mean() * 100, 1),
                     "Dolu": int(df[c].notna().sum())})
    return pd.DataFrame(rows)


def fill_missing(df: pd.DataFrame, cols: list[str], method: str, constant=None) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if method == "mean" and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(out[c].mean())
        elif method == "median" and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(out[c].median())
        elif method == "mode":
            m = out[c].mode()
            if len(m):
                out[c] = out[c].fillna(m.iloc[0])
        elif method == "ffill":
            out[c] = out[c].ffill()
        elif method == "bfill":
            out[c] = out[c].bfill()
        elif method == "interpolate" and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].interpolate()
        elif method == "constant":
            out[c] = out[c].fillna(constant)
    return out


def drop_missing(df: pd.DataFrame, axis: int = 0, subset: list[str] | None = None) -> pd.DataFrame:
    return df.dropna(axis=axis, subset=subset).reset_index(drop=True)


# ---------------- Aykırı değer ----------------

def detect_outliers(series: pd.Series, method: str = "iqr", factor: float = 1.5) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if len(valid) < 4:
        return {"error": "Aykırı değer analizi için yeterli sayısal veri yok."}
    if method == "zscore":
        z = np.abs(stats.zscore(valid))
        mask = pd.Series(False, index=s.index)
        mask.loc[valid.index] = z > factor
        lower = upper = None
    else:  # IQR
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask = (s < lower) | (s > upper)
    n_out = int(mask.sum())
    return {"method": method, "mask": mask.fillna(False), "n_outliers": n_out,
            "pct": round(n_out / len(valid) * 100, 1), "lower": lower, "upper": upper,
            "factor": factor}


def handle_outliers(df: pd.DataFrame, col: str, res: dict, action: str = "cap") -> pd.DataFrame:
    out = df.copy()
    mask = res["mask"]
    if action == "remove":
        return out.loc[~mask].reset_index(drop=True)
    if action == "cap" and res.get("lower") is not None:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(res["lower"], res["upper"])
    elif action == "nan":
        out.loc[mask, col] = np.nan
    return out


# ---------------- Dönüşüm ----------------

TRANSFORMS = {
    "log": "Doğal logaritma ln(x) — sağa çarpık, pozitif veriler için",
    "log1p": "ln(1+x) — sıfır içeren pozitif veriler için",
    "sqrt": "Karekök — hafif çarpıklık, sayım verileri",
    "zscore": "Standartlaştırma (z) — ortalama 0, std 1",
    "minmax": "Min-Max normalizasyon — [0,1] aralığı",
    "boxcox": "Box-Cox — normalliğe en yakın güç dönüşümü (pozitif veri)",
}


def transform(df: pd.DataFrame, col: str, kind: str, new_col: str | None = None) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    s = pd.to_numeric(out[col], errors="coerce")
    name = new_col or f"{col}_{kind}"
    note = ""
    if kind == "log":
        if (s <= 0).any():
            note = "Uyarı: ≤0 değerler NaN oldu (log tanımsız)."
        out[name] = np.log(s.where(s > 0))
    elif kind == "log1p":
        out[name] = np.log1p(s.where(s > -1))
    elif kind == "sqrt":
        out[name] = np.sqrt(s.where(s >= 0))
    elif kind == "zscore":
        out[name] = (s - s.mean()) / s.std(ddof=1)
    elif kind == "minmax":
        rng = s.max() - s.min()
        out[name] = (s - s.min()) / rng if rng else 0.0
    elif kind == "boxcox":
        if (s <= 0).any():
            note = "Box-Cox pozitif veri gerektirir; ≤0 değerler nedeniyle uygulanamadı."
            return out, note
        vals, lam = stats.boxcox(s.dropna())
        res = pd.Series(np.nan, index=s.index)
        res.loc[s.dropna().index] = vals
        out[name] = res
        note = f"Box-Cox λ = {lam:.4f}"
    return out, note


def transform_skew_note(before: pd.Series, after: pd.Series) -> str:
    b = pd.to_numeric(before, errors="coerce").dropna()
    a = pd.to_numeric(after, errors="coerce").dropna()
    if len(b) < 3 or len(a) < 3:
        return ""
    sb = stats.skew(b, bias=False)
    sa = stats.skew(a, bias=False)
    better = abs(sa) < abs(sb)
    return I.bullet(f"Çarpıklık {sb:.3f} → {sa:.3f} "
                    + ("(dağılım normale **yaklaştı** ✓)" if better else "(belirgin iyileşme yok)"))
