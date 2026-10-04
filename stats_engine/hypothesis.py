"""Hipotez testleri: t-testleri, ki-kare, Mann-Whitney, Kruskal-Wallis, oran/z."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import interpret as I


def one_sample_t(series: pd.Series, popmean: float, alpha: float = 0.05, alt: str = "two-sided") -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return {"error": "En az 2 gözlem gerekir."}
    t, p = stats.ttest_1samp(s, popmean, alternative=alt)
    d = (s.mean() - popmean) / s.std(ddof=1) if s.std(ddof=1) else np.nan
    return {"test": "Tek örneklem t-testi", "t": float(t), "p": float(p), "df": int(len(s) - 1),
            "mean": float(s.mean()), "popmean": popmean, "n": int(len(s)),
            "cohens_d": float(d), "alpha": alpha, "alt": alt}


def independent_t(a: pd.Series, b: pd.Series, alpha: float = 0.05, equal_var: bool | None = None,
                  alt: str = "two-sided", labels=("Grup 1", "Grup 2")) -> dict:
    sa = pd.to_numeric(a, errors="coerce").dropna()
    sb = pd.to_numeric(b, errors="coerce").dropna()
    if len(sa) < 2 or len(sb) < 2:
        return {"error": "Her grupta en az 2 gözlem gerekir."}
    # Varyans homojenliği (Levene) — equal_var otomatik seç
    lev_stat, lev_p = stats.levene(sa, sb, center="median")
    if equal_var is None:
        equal_var = lev_p >= alpha
    t, p = stats.ttest_ind(sa, sb, equal_var=equal_var, alternative=alt)
    # Cohen's d (pooled)
    n1, n2 = len(sa), len(sb)
    sp = np.sqrt(((n1 - 1) * sa.var(ddof=1) + (n2 - 1) * sb.var(ddof=1)) / (n1 + n2 - 2))
    d = (sa.mean() - sb.mean()) / sp if sp else np.nan
    return {"test": "Bağımsız örneklem t-testi" + (" (Welch)" if not equal_var else ""),
            "t": float(t), "p": float(p), "n1": n1, "n2": n2,
            "mean1": float(sa.mean()), "mean2": float(sb.mean()),
            "levene_p": float(lev_p), "equal_var": bool(equal_var),
            "cohens_d": float(d), "alpha": alpha, "alt": alt, "labels": labels}


def paired_t(a: pd.Series, b: pd.Series, alpha: float = 0.05, alt: str = "two-sided",
             labels=("Öncesi", "Sonrası")) -> dict:
    pair = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(pair) < 2:
        return {"error": "En az 2 eşleşmiş çift gerekir."}
    x, y = pair.iloc[:, 0], pair.iloc[:, 1]
    t, p = stats.ttest_rel(x, y, alternative=alt)
    diff = x - y
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else np.nan
    return {"test": "Eşleştirilmiş örneklem t-testi", "t": float(t), "p": float(p),
            "n": int(len(pair)), "mean_diff": float(diff.mean()),
            "mean1": float(x.mean()), "mean2": float(y.mean()),
            "cohens_d": float(d), "alpha": alpha, "alt": alt, "labels": labels}


def mann_whitney(a: pd.Series, b: pd.Series, alpha: float = 0.05, alt: str = "two-sided") -> dict:
    sa = pd.to_numeric(a, errors="coerce").dropna()
    sb = pd.to_numeric(b, errors="coerce").dropna()
    if len(sa) < 1 or len(sb) < 1:
        return {"error": "Yeterli gözlem yok."}
    u, p = stats.mannwhitneyu(sa, sb, alternative=alt)
    return {"test": "Mann-Whitney U (parametrik olmayan)", "U": float(u), "p": float(p),
            "median1": float(sa.median()), "median2": float(sb.median()), "alpha": alpha}


def kruskal(groups: list[pd.Series], labels: list[str], alpha: float = 0.05) -> dict:
    arrs = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
    arrs = [a for a in arrs if len(a) > 0]
    if len(arrs) < 2:
        return {"error": "En az 2 grup gerekir."}
    h, p = stats.kruskal(*arrs)
    return {"test": "Kruskal-Wallis H (parametrik olmayan ANOVA)", "H": float(h), "p": float(p),
            "k": len(arrs), "labels": labels, "alpha": alpha}


def wilcoxon_signed(a: pd.Series, b: pd.Series, alpha: float = 0.05, alt: str = "two-sided",
                    labels=("Öncesi", "Sonrası")) -> dict:
    pair = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(pair) < 5:
        return {"error": "Wilcoxon için en az ~5 eşleşmiş çift önerilir."}
    x, y = pair.iloc[:, 0], pair.iloc[:, 1]
    try:
        w, p = stats.wilcoxon(x, y, alternative=alt)
    except Exception as e:
        return {"error": f"Wilcoxon hesaplanamadı: {e}"}
    return {"test": "Wilcoxon işaretli sıra testi (parametrik olmayan eşleştirilmiş)", "W": float(w),
            "p": float(p), "n": int(len(pair)), "median_diff": float((x - y).median()),
            "median1": float(x.median()), "median2": float(y.median()), "alpha": alpha, "labels": labels}


def friedman(groups: list[pd.Series], labels: list[str], alpha: float = 0.05) -> dict:
    df_all = pd.concat([pd.to_numeric(g, errors="coerce").reset_index(drop=True) for g in groups], axis=1).dropna()
    if df_all.shape[1] < 3:
        return {"error": "Friedman testi için en az 3 tekrarlı ölçüm gerekir."}
    if len(df_all) < 2:
        return {"error": "Yeterli gözlem yok."}
    stat, p = stats.friedmanchisquare(*[df_all.iloc[:, i] for i in range(df_all.shape[1])])
    return {"test": "Friedman testi (parametrik olmayan tekrarlı ölçüm ANOVA)", "chi2": float(stat),
            "p": float(p), "k": df_all.shape[1], "n": int(len(df_all)), "labels": labels, "alpha": alpha}


def chi_square(df: pd.DataFrame, col_a: str, col_b: str, alpha: float = 0.05) -> dict:
    table = pd.crosstab(df[col_a], df[col_b])
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return {"error": "Ki-kare için en az 2x2 çapraz tablo gerekir."}
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.values.sum()
    min_dim = min(table.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan
    return {"test": "Ki-kare bağımsızlık testi", "chi2": float(chi2), "p": float(p), "dof": int(dof),
            "cramer_v": float(cramer_v), "table": table,
            "expected": pd.DataFrame(expected, index=table.index, columns=table.columns),
            "min_expected": float(np.min(expected)), "alpha": alpha}


def fisher_exact(df: pd.DataFrame, col_a: str, col_b: str, alpha: float = 0.05) -> dict:
    table = pd.crosstab(df[col_a], df[col_b])
    if table.shape != (2, 2):
        return {"error": "Fisher exact yalnızca 2×2 tablo içindir (her değişkende tam 2 kategori)."}
    odds, p = stats.fisher_exact(table.values)
    return {"test": "Fisher kesin (exact) testi", "odds": float(odds), "p": float(p),
            "table": table, "alpha": alpha}


def mcnemar_test(df: pd.DataFrame, col_a: str, col_b: str, alpha: float = 0.05) -> dict:
    """Eşleştirilmiş ikili kategorik (öncesi/sonrası) — McNemar."""
    from statsmodels.stats.contingency_tables import mcnemar
    table = pd.crosstab(df[col_a], df[col_b])
    if table.shape != (2, 2):
        return {"error": "McNemar 2×2 eşleştirilmiş tablo gerektirir."}
    res = mcnemar(table.values, exact=True)
    return {"test": "McNemar testi (eşleştirilmiş kategorik)", "stat": float(res.statistic),
            "p": float(res.pvalue), "table": table, "alpha": alpha}


def chi2_goodness(observed: list[float], alpha: float = 0.05, expected: list[float] | None = None,
                  labels: list[str] | None = None) -> dict:
    obs = np.asarray(observed, float)
    if expected is None:
        exp = np.full_like(obs, obs.sum() / len(obs))  # eşit dağılım
    else:
        exp = np.asarray(expected, float)
        exp = exp / exp.sum() * obs.sum()  # gözlem toplamına ölçekle
    chi2, p = stats.chisquare(obs, exp)
    return {"test": "Ki-kare uyum iyiliği testi", "chi2": float(chi2), "p": float(p),
            "dof": len(obs) - 1, "observed": obs, "expected": exp,
            "labels": labels or [f"K{i+1}" for i in range(len(obs))], "alpha": alpha}


def point_biserial(binary: pd.Series, continuous: pd.Series, alpha: float = 0.05) -> dict:
    b = pd.to_numeric(binary, errors="coerce")
    c = pd.to_numeric(continuous, errors="coerce")
    pair = pd.concat([b, c], axis=1).dropna()
    if pair.iloc[:, 0].nunique() != 2:
        return {"error": "İlk değişken tam 2 düzeyli (0/1) olmalı."}
    r, p = stats.pointbiserialr(pair.iloc[:, 0], pair.iloc[:, 1])
    return {"test": "Nokta-çift serili korelasyon", "r": float(r), "p": float(p),
            "n": int(len(pair)), "alpha": alpha}


def binomial_test(successes: int, n: int, p0: float = 0.5, alpha: float = 0.05,
                  alt: str = "two-sided") -> dict:
    res = stats.binomtest(successes, n, p0, alternative=alt)
    return {"test": "Binom testi (tek oran, kesin)", "successes": successes, "n": n,
            "prop": successes / n if n else np.nan, "p0": p0, "p": float(res.pvalue),
            "ci": res.proportion_ci(1 - alpha), "alpha": alpha}


def proportions_z(counts, nobs, alpha: float = 0.05, alt: str = "two-sided", value=None) -> dict:
    from statsmodels.stats.proportion import proportions_ztest
    stat, p = proportions_ztest(counts, nobs, value=value, alternative=alt)
    two = hasattr(counts, "__len__")
    return {"test": "İki oran z-testi" if two else "Tek oran z-testi", "z": float(stat),
            "p": float(p), "alpha": alpha,
            "props": [c / n for c, n in zip(counts, nobs)] if two else [counts / nobs]}


def bartlett_test(groups: list[pd.Series], labels: list[str], alpha: float = 0.05) -> dict:
    arrs = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
    arrs = [a for a in arrs if len(a) > 1]
    if len(arrs) < 2:
        return {"error": "Bartlett için en az 2 grup gerekir."}
    stat, p = stats.bartlett(*arrs)
    return {"test": "Bartlett varyans homojenliği testi", "stat": float(stat), "p": float(p),
            "k": len(arrs), "alpha": alpha}


# ---- yorumlama ----

def interpret(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    t = res["test"]
    lines = [f"### {t}"]
    if "t" in res:
        lines.append(I.bullet(f"t = {res['t']:.3f}, sd = {res.get('df', res.get('n',0)-1)}, p = {res['p']:.4g} {I.stars(res['p'])}"))
        if "mean1" in res and "mean2" in res:
            lines.append(I.bullet(f"Ortalamalar: {res.get('labels',('1','2'))[0]} = {res['mean1']:.4f}, "
                                  f"{res.get('labels',('1','2'))[1]} = {res['mean2']:.4f}"))
        if "popmean" in res:
            lines.append(I.bullet(f"Örneklem ortalaması = {res['mean']:.4f} vs. varsayılan µ₀ = {res['popmean']}"))
        if "levene_p" in res:
            lines.append(I.bullet(f"Levene p = {res['levene_p']:.4g} → varyanslar "
                                  + ("homojen" if res['equal_var'] else "homojen değil, Welch düzeltmesi uygulandı")))
        if "cohens_d" in res and res["cohens_d"] == res["cohens_d"]:
            lines.append(I.bullet(f"Etki büyüklüğü Cohen's d = {res['cohens_d']:.3f} ({I.cohens_d_label(res['cohens_d'])})"))
    elif "U" in res:
        lines.append(I.bullet(f"U = {res['U']:.1f}, p = {res['p']:.4g} {I.stars(res['p'])}"))
        lines.append(I.bullet(f"Medyanlar: {res['median1']:.4f} vs {res['median2']:.4f}"))
    elif "W" in res:
        lines.append(I.bullet(f"W = {res['W']:.1f}, n = {res['n']}, p = {res['p']:.4g} {I.stars(res['p'])}"))
        lines.append(I.bullet(f"Medyanlar: {res.get('labels',('1','2'))[0]} = {res['median1']:.4f}, "
                              f"{res.get('labels',('1','2'))[1]} = {res['median2']:.4f} "
                              f"(medyan fark = {res['median_diff']:.4f})"))
    elif "chi2" in res and "k" in res:  # Friedman
        lines.append(I.bullet(f"χ² = {res['chi2']:.3f}, k = {res['k']} ölçüm, n = {res['n']}, "
                              f"p = {res['p']:.4g} {I.stars(res['p'])}"))
    elif "odds" in res:  # Fisher
        lines.append(I.bullet(f"Odds oranı = {res['odds']:.3f}, p = {res['p']:.4g} {I.stars(res['p'])}"))
        lines.append(I.bullet("Küçük örneklem/beklenen frekans için ki-kareye göre daha güvenilirdir."))
    elif "successes" in res:  # Binom
        lines.append(I.bullet(f"Gözlenen oran = {res['prop']:.3f} ({res['successes']}/{res['n']}) vs. p₀ = {res['p0']}, "
                              f"p = {res['p']:.4g} {I.stars(res['p'])}"))
        lines.append(I.bullet(f"%{int((1-a)*100)} güven aralığı: [{res['ci'].low:.3f}, {res['ci'].high:.3f}]"))
    elif "z" in res and "props" in res:  # Oran z-testi
        lines.append(I.bullet(f"z = {res['z']:.3f}, p = {res['p']:.4g} {I.stars(res['p'])}"))
        lines.append(I.bullet("Oranlar: " + ", ".join(f"{p:.3f}" for p in res["props"])))
    elif "r" in res:  # nokta-çift serili
        lines.append(I.bullet(f"r_pb = {res['r']:.3f}, n = {res['n']}, p = {res['p']:.4g} {I.stars(res['p'])}"))
        lines.append(I.bullet(f"İlişki **{I.effect_size_r(res['r'])}** ({I.direction(res['r'])})."))
    elif "observed" in res:  # ki-kare uyum iyiliği
        lines.append(I.bullet(f"χ² = {res['chi2']:.3f}, sd = {res['dof']}, p = {res['p']:.4g} {I.stars(res['p'])}"))
        lines.append(I.bullet("Gözlenen dağılımın beklenenden sapıp sapmadığını sınar."))
    elif "stat" in res and "table" in res:  # McNemar
        lines.append(I.bullet(f"İstatistik = {res['stat']:.3f}, p = {res['p']:.4g} {I.stars(res['p'])}"))
        lines.append(I.bullet("Eşleştirilmiş ikili sonuçlarda değişimin yönünü/anlamlılığını sınar."))
    elif "stat" in res and "k" in res:  # Bartlett
        lines.append(I.bullet(f"Bartlett χ² = {res['stat']:.3f}, k = {res['k']} grup, p = {res['p']:.4g} {I.stars(res['p'])}"))
        lines.append(I.bullet("p < α → varyanslar homojen değil (normal dağılıma duyarlı; Levene daha dayanıklıdır)."))
    elif "H" in res:
        lines.append(I.bullet(f"H = {res['H']:.3f}, k = {res['k']} grup, p = {res['p']:.4g} {I.stars(res['p'])}"))
    elif "chi2" in res:
        lines.append(I.bullet(f"χ² = {res['chi2']:.3f}, sd = {res['dof']}, p = {res['p']:.4g} {I.stars(res['p'])}"))
        lines.append(I.bullet(f"Cramér's V = {res['cramer_v']:.3f} (ilişki gücü)"))
        if res["min_expected"] < 5:
            lines.append(I.bullet(f"⚠️ En küçük beklenen frekans = {res['min_expected']:.2f} < 5; "
                                  "ki-kare varsayımı zayıf, Fisher exact testi düşünülebilir."))
    lines.append("")
    lines.append("**Karar:** " + I.p_sentence(res["p"], a))
    return I.joinlines(lines)
