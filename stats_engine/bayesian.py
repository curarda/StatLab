"""Bayesçi analiz: Bayes faktörü (t-testi, korelasyon) ve Bayesçi A/B (Beta-Binom)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import interpret as I


def bf_label(bf10: float) -> str:
    """Jeffreys ölçeğinde Bayes faktörü yorumu (BF10)."""
    bf = bf10
    if bf < 1:
        inv = 1 / bf if bf else np.inf
        base = ("anecdotal (zayıf)" if inv < 3 else "orta" if inv < 10 else "güçlü" if inv < 30
                else "çok güçlü" if inv < 100 else "kesin")
        return f"H0 lehine {base} kanıt (BF01={inv:.1f})"
    base = ("anlamsız/çok zayıf" if bf < 3 else "orta" if bf < 10 else "güçlü" if bf < 30
            else "çok güçlü" if bf < 100 else "kesin (decisive)")
    return f"H1 lehine {base} kanıt"


def bayesian_ttest(a: pd.Series, b: pd.Series = None, paired: bool = False, mu: float = 0.0,
                   alpha: float = 0.05) -> dict:
    """Bayesçi t-testi: JZS Bayes faktörü + fark için sonsal olasılık."""
    import pingouin as pg
    sa = pd.to_numeric(a, errors="coerce").dropna()
    if b is not None:
        sb = pd.to_numeric(b, errors="coerce").dropna()
        if paired:
            pair = pd.concat([sa, sb], axis=1).dropna()
            res = pg.ttest(pair.iloc[:, 0], pair.iloc[:, 1], paired=True)
            diff = pair.iloc[:, 0] - pair.iloc[:, 1]
            mean_diff, se = diff.mean(), diff.sem()
        else:
            res = pg.ttest(sa, sb, paired=False)
            mean_diff = sa.mean() - sb.mean()
            se = np.sqrt(sa.var(ddof=1) / len(sa) + sb.var(ddof=1) / len(sb))
        label = "Bayesçi bağımsız t-testi" if not paired else "Bayesçi eşleştirilmiş t-testi"
    else:
        res = pg.ttest(sa, mu)
        mean_diff = sa.mean() - mu
        se = sa.sem()
        label = "Bayesçi tek örneklem t-testi"
    bf10 = float(res["BF10"].iloc[0])
    # Sonsal (yaklaşık, düz önsel): fark ~ Normal(mean_diff, se)
    p_gt0 = float(stats.norm.cdf(mean_diff / se)) if se else np.nan
    zc = stats.norm.ppf(1 - alpha / 2)
    ci = (float(mean_diff - zc * se), float(mean_diff + zc * se))
    return {"test": label, "bf10": bf10, "mean_diff": float(mean_diff), "se": float(se),
            "p_gt0": p_gt0, "ci": ci, "cohens_d": float(res["cohen_d"].iloc[0]) if "cohen_d" in res else np.nan,
            "p_freq": float(res["p_val"].iloc[0]), "alpha": alpha}


def bayesian_correlation(x: pd.Series, y: pd.Series, alpha: float = 0.05) -> dict:
    import pingouin as pg
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    n = len(pair)
    bf10 = float(pg.bayesfactor_pearson(r, n))
    return {"test": "Bayesçi Pearson korelasyonu", "r": float(r), "n": n, "bf10": bf10, "alpha": alpha}


def bayesian_ab(s_a: int, n_a: int, s_b: int, n_b: int, n_samples: int = 100000,
                alpha: float = 0.05) -> dict:
    """Bayesçi A/B (Beta-Binom eşlenik). P(B>A) ve beklenen kayıp."""
    rng = np.random.default_rng(42)
    post_a = rng.beta(1 + s_a, 1 + n_a - s_a, n_samples)
    post_b = rng.beta(1 + s_b, 1 + n_b - s_b, n_samples)
    prob_b = float(np.mean(post_b > post_a))
    lift = post_b - post_a
    expected_lift = float(np.mean(lift))
    # kayıp fonksiyonu: B seçilirse beklenen kaçırılan dönüşüm
    loss_choose_b = float(np.mean(np.maximum(post_a - post_b, 0)))
    ci = (float(np.percentile(lift, alpha / 2 * 100)), float(np.percentile(lift, (1 - alpha / 2) * 100)))
    return {"prob_b_better": prob_b, "expected_lift": expected_lift, "ci": ci,
            "p_a": s_a / n_a, "p_b": s_b / n_b, "exp_loss_b": loss_choose_b, "alpha": alpha}


def interpret_bayes_ttest(res: dict) -> str:
    lines = [f"### {res['test']}"]
    lines.append(I.bullet(f"**Bayes faktörü BF₁₀ = {res['bf10']:.3g}** → {bf_label(res['bf10'])}."))
    lines.append(I.bullet(f"Ortalama fark = {res['mean_diff']:.4f}, %{int((1-res['alpha'])*100)} güven aralığı "
                          f"[{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]."))
    if res["p_gt0"] == res["p_gt0"]:
        lines.append(I.bullet(f"Sonsal olasılık P(fark > 0) ≈ **%{res['p_gt0']*100:.1f}** "
                              "(düz önsel yaklaşık)."))
    lines.append(I.bullet(f"Karşılaştırma: klasik p-değeri = {res['p_freq']:.4g}. "
                          "Bayes faktörü, klasik testin aksine **H0 lehine de kanıt** sunabilir (etki YOK diyebilir)."))
    lines.append(I.bullet("BF₁₀ > 3 kayda değer, > 10 güçlü kanıt; 1/3 – 3 arası ise 'kararsız' bölgedir "
                          "(veri henüz ayırt edici değil)."))
    return I.joinlines(lines)


def interpret_bayes_corr(res: dict) -> str:
    lines = [f"### {res['test']}"]
    lines.append(I.bullet(f"r = {res['r']:.4f} (n={res['n']}), **BF₁₀ = {res['bf10']:.3g}** → {bf_label(res['bf10'])}."))
    lines.append(I.bullet(f"İlişki gücü: {I.effect_size_r(res['r'])} ({I.direction(res['r'])})."))
    return I.joinlines(lines)


def interpret_bayes_ab(res: dict) -> str:
    lines = ["### Bayesçi A/B testi (Beta-Binom)"]
    lines.append(I.bullet(f"Kontrol A: %{res['p_a']*100:.2f} · Varyant B: %{res['p_b']*100:.2f}"))
    lines.append(I.bullet(f"**P(B > A) = %{res['prob_b_better']*100:.1f}** → "
                          + ("B'nin daha iyi olduğuna dair güçlü kanıt." if res['prob_b_better'] > 0.95
                             else "B lehine eğilim ama kesin değil." if res['prob_b_better'] > 0.8
                             else "belirgin bir üstünlük yok.")))
    lines.append(I.bullet(f"Beklenen fark (B−A): {res['expected_lift']*100:+.2f} puan, "
                          f"%{int((1-res['alpha'])*100)} güvenilir aralık "
                          f"[{res['ci'][0]*100:+.2f}, {res['ci'][1]*100:+.2f}]."))
    lines.append(I.bullet(f"B'yi seçmenin beklenen kaybı (risk): %{res['exp_loss_b']*100:.3f} puan. "
                          "Düşükse B'ye geçmek güvenlidir."))
    lines.append(I.bullet("Bayesçi yaklaşım 'kaç kez baktığından' etkilenmez; sonucu doğrudan olasılık olarak okursun "
                          "(p-değeri gibi dolaylı değil)."))
    return I.joinlines(lines)
