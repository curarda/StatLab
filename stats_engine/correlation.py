"""Korelasyon analizi: Pearson ve Spearman + anlamlılık."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import interpret as I


def correlation_matrix(df: pd.DataFrame, columns: list[str], method: str = "pearson"):
    """Korelasyon ve p-değeri matrislerini döndür."""
    cols = columns
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    n = len(cols)
    corr = pd.DataFrame(np.eye(n), index=cols, columns=cols)
    pmat = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    for i in range(n):
        for j in range(n):
            if i == j:
                corr.iloc[i, j] = 1.0
                pmat.iloc[i, j] = 0.0
                continue
            pair = data[[cols[i], cols[j]]].dropna()
            if len(pair) < 3:
                corr.iloc[i, j] = np.nan
                pmat.iloc[i, j] = np.nan
                continue
            x, y = pair[cols[i]], pair[cols[j]]
            if method == "spearman":
                r, p = stats.spearmanr(x, y)
            elif method == "kendall":
                r, p = stats.kendalltau(x, y)
            else:
                r, p = stats.pearsonr(x, y)
            corr.iloc[i, j] = r
            pmat.iloc[i, j] = p
    return corr, pmat


def pairwise(df: pd.DataFrame, x: str, y: str, method: str = "pearson", alpha: float = 0.05) -> dict:
    pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        return {"error": "Korelasyon için en az 3 eşleşmiş gözlem gerekir."}
    xv, yv = pair[x], pair[y]
    if method == "spearman":
        r, p = stats.spearmanr(xv, yv)
        label = "Spearman ρ"
    elif method == "kendall":
        r, p = stats.kendalltau(xv, yv)
        label = "Kendall τ"
    else:
        r, p = stats.pearsonr(xv, yv)
        label = "Pearson r"
    n = len(pair)
    out = {"x": x, "y": y, "method": method, "label": label, "r": float(r),
           "p": float(p), "n": n, "r2": float(r * r), "alpha": alpha}
    # Fisher z güven aralığı (Pearson için)
    if method == "pearson" and abs(r) < 1 and n > 3:
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        zc = stats.norm.ppf(1 - alpha / 2)
        out["ci"] = (float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se)))
    return out


def interpret_pairwise(res: dict) -> str:
    if "error" in res:
        return res["error"]
    r, p, alpha = res["r"], res["p"], res["alpha"]
    lines = [f"**{res['x']} ↔ {res['y']}** ({res['label']}):"]
    lines.append(I.bullet(f"Katsayı = {r:.4f} {I.stars(p)}  |  n = {res['n']}  |  R² = {res['r2']:.4f}"))
    if "ci" in res:
        lines.append(I.bullet(f"%{int((1-alpha)*100)} güven aralığı: [{res['ci'][0]:.3f}, {res['ci'][1]:.3f}]"))
    lines.append(I.bullet(
        f"İlişki **{I.effect_size_r(r)}** ve **{I.direction(r)}**. "
        f"Belirlilik katsayısı R²={res['r2']:.3f}, yani bir değişken diğerindeki değişimin "
        f"yaklaşık %{res['r2']*100:.1f}'ini paylaşıyor."))
    lines.append(I.bullet(I.p_sentence(p, alpha)))
    if p < alpha:
        lines.append(I.bullet(
            f"Yorum: İki değişken arasında {I.direction(r).split()[0]} yönlü, {I.effect_size_r(r)} "
            "ve anlamlı bir doğrusal ilişki var. **Ancak korelasyon nedensellik değildir.**"))
    else:
        lines.append(I.bullet("Yorum: Doğrusal bir ilişkiye dair yeterli istatistiksel kanıt yok."))
    return I.joinlines(lines)


def partial(df: pd.DataFrame, x: str, y: str, covars: list[str], method: str = "pearson",
            alpha: float = 0.05) -> dict:
    """Kısmi korelasyon: covars değişkenlerinin etkisi kontrol edilerek x-y ilişkisi."""
    import pingouin as pg
    cols = [x, y] + covars
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < len(covars) + 4:
        return {"error": "Kısmi korelasyon için yeterli veri yok."}
    try:
        res = pg.partial_corr(data, x=x, y=y, covar=covars, method=method)
    except Exception as e:
        return {"error": f"Kısmi korelasyon hesaplanamadı: {e}"}
    r = res.iloc[0]
    # ham (sıfırıncı derece) korelasyon karşılaştırma için
    raw = pairwise(df, x, y, method, alpha)
    return {"x": x, "y": y, "covars": covars, "r": float(r["r"]), "p": float(r["p_val"]),
            "n": int(r["n"]), "ci": r["CI95"], "raw_r": raw.get("r"), "method": method, "alpha": alpha}


def interpret_partial(res: dict) -> str:
    if "error" in res:
        return res["error"]
    r, p, a = res["r"], res["p"], res["alpha"]
    lines = [f"### Kısmi korelasyon — {res['x']} ↔ {res['y']} | kontrol: {', '.join(res['covars'])}"]
    lines.append(I.bullet(f"Kısmi r = {r:.4f} {I.stars(p)} (n={res['n']}, p={p:.4g}). Ham r = {res['raw_r']:.4f}."))
    diff = abs(res["raw_r"]) - abs(r)
    if diff > 0.1:
        lines.append(I.bullet(f"Kontrol değişken(ler)i çıkarınca ilişki **belirgin zayıfladı** ({res['raw_r']:.2f} → {r:.2f}) "
                              "→ ilişkinin önemli bölümü kontrol değişkeni üzerinden (olası sahte/dolaylı ilişki)."))
    elif diff < -0.05:
        lines.append(I.bullet("Kontrol sonrası ilişki **güçlendi** → bastırıcı (suppressor) etki olabilir."))
    else:
        lines.append(I.bullet("Kontrol değişkeni ilişkiyi pek değiştirmedi → ilişki büyük ölçüde doğrudan."))
    lines.append(I.bullet(f"İlişki gücü: {I.effect_size_r(r)} ({I.direction(r)}). " + I.p_sentence(p, a)))
    return I.joinlines(lines)


def top_correlations(corr: pd.DataFrame, pmat: pd.DataFrame, alpha: float = 0.05, top: int = 8) -> pd.DataFrame:
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            p = pmat.iloc[i, j]
            if pd.isna(r):
                continue
            rows.append({"Değişken 1": cols[i], "Değişken 2": cols[j], "r": r, "p": p,
                         "Anlamlılık": I.stars(p) or "—", "Güç": I.effect_size_r(r)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.reindex(out["r"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)
    return out
