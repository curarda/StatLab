"""Karışık-tip ilişki matrisi: sayısal+kategorik değişkenler tek ölçekte (0–1).

- sayısal ↔ sayısal : |Pearson r|
- kategorik ↔ kategorik : Cramér's V (bias düzeltmeli)
- sayısal ↔ kategorik : korelasyon oranı η (ANOVA'dan)
Tümü 0 (ilişki yok) – 1 (tam ilişki) aralığında, tek ısı haritasında karşılaştırılabilir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import interpret as I


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 2


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    tab = pd.crosstab(a, b)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(tab)[0]
    n = tab.values.sum()
    phi2 = chi2 / n
    r, k = tab.shape
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else np.nan


def correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    """η — kategorik değişkenin sayısal değişkendeki varyansı açıklama oranı (0–1)."""
    d = pd.concat([cat, pd.to_numeric(num, errors="coerce")], axis=1).dropna()
    d.columns = ["c", "n"]
    if d["c"].nunique() < 2 or len(d) < 3:
        return np.nan
    grand = d["n"].mean()
    ss_b = sum(len(g) * (g["n"].mean() - grand) ** 2 for _, g in d.groupby("c"))
    ss_t = ((d["n"] - grand) ** 2).sum()
    return float(np.sqrt(ss_b / ss_t)) if ss_t else np.nan


def association_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(cols)
    M = pd.DataFrame(np.eye(n), index=cols, columns=cols)
    T = pd.DataFrame("", index=cols, columns=cols)  # yöntem
    for i in range(n):
        for j in range(i + 1, n):
            a, b = df[cols[i]], df[cols[j]]
            an, bn = _is_num(a), _is_num(b)
            if an and bn:
                pair = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
                v = abs(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if len(pair) > 2 else np.nan
                meth = "|r|"
            elif not an and not bn:
                v = cramers_v(a, b)
                meth = "V"
            else:
                cat, num = (a, b) if not an else (b, a)
                v = correlation_ratio(cat, num)
                meth = "η"
            M.iloc[i, j] = M.iloc[j, i] = v
            T.iloc[i, j] = T.iloc[j, i] = meth
    return M, T


def top_associations(M: pd.DataFrame, T: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    rows = []
    cols = list(M.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = M.iloc[i, j]
            if pd.isna(v):
                continue
            rows.append({"Değişken 1": cols[i], "Değişken 2": cols[j],
                         "İlişki (0-1)": v, "Yöntem": T.iloc[i, j],
                         "Güç": _label(v)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("İlişki (0-1)", ascending=False).head(top).reset_index(drop=True)
    return out


def _label(v: float) -> str:
    if v != v:
        return "—"
    return ("çok güçlü" if v >= 0.7 else "güçlü" if v >= 0.5 else "orta" if v >= 0.3
            else "zayıf" if v >= 0.1 else "ihmal edilebilir")


def interpret_association(top: pd.DataFrame) -> str:
    lines = ["### Karışık-tip ilişki matrisi — yorum katmanı"]
    lines.append(I.bullet("Yöntem: sayısal–sayısal **|r|**, kategorik–kategorik **Cramér's V**, "
                          "sayısal–kategorik **korelasyon oranı η**. Hepsi 0–1; doğrudan kıyaslanabilir."))
    if top.empty:
        lines.append(I.bullet("Belirgin ilişki bulunamadı."))
        return I.joinlines(lines)
    strong = top[top["İlişki (0-1)"] >= 0.3]
    if not strong.empty:
        r0 = strong.iloc[0]
        lines.append(I.bullet(f"En güçlü ilişki: **{r0['Değişken 1']} ↔ {r0['Değişken 2']}** "
                              f"({r0['Yöntem']}={r0['İlişki (0-1)']:.2f}, {r0['Güç']})."))
        lines.append(I.bullet(f"Toplam {len(strong)} orta/güçlü ilişki var — bunlar modelleme ve segmentasyon için "
                              "en umut verici değişken çiftleri."))
    else:
        lines.append(I.bullet("İlişkiler zayıf; değişkenler büyük ölçüde bağımsız görünüyor."))
    lines.append(I.bullet("Not: η ve V yönsüzdür (yalnızca güç); yön için sayısal çiftlerde işaretli korelasyona bakın."))
    return I.joinlines(lines)
