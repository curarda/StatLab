"""Betimsel istatistik ve normallik testleri."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import interpret as I


def describe(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append({
            "Değişken": c,
            "N": int(s.count()),
            "Ortalama": s.mean(),
            "Std. Sapma": s.std(ddof=1),
            "Std. Hata": s.sem(),
            "Min": s.min(),
            "%25": s.quantile(0.25),
            "Medyan": s.median(),
            "%75": s.quantile(0.75),
            "Max": s.max(),
            "Çarpıklık": stats.skew(s, bias=False) if len(s) > 2 else np.nan,
            "Basıklık": stats.kurtosis(s, bias=False) if len(s) > 3 else np.nan,
            "VK %": (s.std(ddof=1) / s.mean() * 100) if s.mean() != 0 else np.nan,
        })
    return pd.DataFrame(rows)


def normality(series: pd.Series, alpha: float = 0.05) -> dict:
    """Shapiro-Wilk (n<5000) ve D'Agostino K² normallik testleri."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    out = {"n": int(len(s)), "skew": float(stats.skew(s, bias=False)) if len(s) > 2 else np.nan,
           "kurtosis": float(stats.kurtosis(s, bias=False)) if len(s) > 3 else np.nan}
    if len(s) < 3:
        out["error"] = "Normallik testi için en az 3 gözlem gerekir."
        return out
    # Shapiro
    if len(s) <= 5000:
        w, p = stats.shapiro(s)
        out["shapiro_W"] = float(w)
        out["shapiro_p"] = float(p)
    # D'Agostino
    if len(s) >= 8:
        k2, p2 = stats.normaltest(s)
        out["dagostino_K2"] = float(k2)
        out["dagostino_p"] = float(p2)
    # Karar: mevcut testlerden en muhafazakârı
    ps = [out.get("shapiro_p"), out.get("dagostino_p")]
    ps = [p for p in ps if p is not None and p == p]
    out["p"] = float(min(ps)) if ps else np.nan
    out["normal"] = bool(out["p"] >= alpha) if ps else None
    return out


def interpret_normality(name: str, res: dict, alpha: float = 0.05) -> str:
    if "error" in res:
        return f"**{name}** — {res['error']}"
    lines = [f"**{name}** normallik değerlendirmesi (n={res['n']}):"]
    if "shapiro_p" in res:
        lines.append(I.bullet(f"Shapiro-Wilk: W={res['shapiro_W']:.4f}, p={res['shapiro_p']:.4f}"))
    if "dagostino_p" in res:
        lines.append(I.bullet(f"D'Agostino K²={res['dagostino_K2']:.4f}, p={res['dagostino_p']:.4f}"))
    lines.append(I.bullet(f"Çarpıklık={res['skew']:.3f}, Basıklık={res['kurtosis']:.3f}"))
    if res.get("normal") is True:
        lines.append(I.bullet(
            f"p ≥ {alpha:g}: Dağılımın normalden anlamlı biçimde saptığına dair kanıt yok; "
            "**normal dağılım varsayımı korunabilir.** Parametrik testler (t-testi, Pearson, ANOVA) uygundur."))
    elif res.get("normal") is False:
        lines.append(I.bullet(
            f"p < {alpha:g}: Dağılım normalden **anlamlı biçimde sapıyor.** "
            "Parametrik test varsayımları zedelenebilir; dönüşüm (log vb.) veya parametrik olmayan "
            "alternatifler (Spearman, Mann-Whitney, Kruskal-Wallis) düşünülmeli."))
    return I.joinlines(lines)
