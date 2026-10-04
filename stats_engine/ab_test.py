"""A/B test analizi ve deney tasarımı (örneklem büyüklüğü)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize, confint_proportions_2indep
from statsmodels.stats.power import NormalIndPower

from . import interpret as I


def design_conversion(baseline: float, mde: float, power: float = 0.80, alpha: float = 0.05,
                      relative: bool = True, daily_traffic: int | None = None,
                      alt: str = "two-sided") -> dict:
    """Dönüşüm (oran) A/B testi için gereken örneklem büyüklüğü."""
    p1 = baseline
    p2 = baseline * (1 + mde) if relative else baseline + mde
    p2 = min(max(p2, 1e-6), 1 - 1e-6)
    effect = abs(proportion_effectsize(p1, p2))
    n = NormalIndPower().solve_power(effect_size=effect, power=power, alpha=alpha, alternative=alt)
    n = int(np.ceil(n))
    out = {"p1": p1, "p2": p2, "mde": mde, "relative": relative, "power": power, "alpha": alpha,
           "n_per_arm": n, "n_total": 2 * n, "effect": float(effect)}
    if daily_traffic:
        out["days"] = int(np.ceil(2 * n / daily_traffic))
        out["daily_traffic"] = daily_traffic
    return out


def analyze_conversion(s_a: int, n_a: int, s_b: int, n_b: int, alpha: float = 0.05,
                       alt: str = "two-sided") -> dict:
    """İki kollu dönüşüm A/B testi (kontrol A vs varyant B)."""
    p_a, p_b = s_a / n_a, s_b / n_b
    z, p = proportions_ztest([s_b, s_a], [n_b, n_a], alternative=alt)
    ci_low, ci_upp = confint_proportions_2indep(s_b, n_b, s_a, n_a, method="wald", compare="diff", alpha=alpha)
    abs_lift = p_b - p_a
    rel_lift = abs_lift / p_a if p_a else np.nan
    return {"p_a": p_a, "p_b": p_b, "n_a": n_a, "n_b": n_b, "s_a": s_a, "s_b": s_b,
            "z": float(z), "p": float(p), "abs_lift": float(abs_lift), "rel_lift": float(rel_lift),
            "ci": (float(ci_low), float(ci_upp)), "alpha": alpha, "kind": "conversion"}


def analyze_continuous(a: pd.Series, b: pd.Series, alpha: float = 0.05, alt: str = "two-sided",
                       labels=("A (kontrol)", "B (varyant)")) -> dict:
    """Sürekli metrik (gelir, süre vb.) için A/B testi."""
    sa = pd.to_numeric(a, errors="coerce").dropna()
    sb = pd.to_numeric(b, errors="coerce").dropna()
    lev_p = stats.levene(sa, sb, center="median")[1]
    equal_var = lev_p >= alpha
    t, p = stats.ttest_ind(sa, sb, equal_var=equal_var, alternative=alt)
    n1, n2 = len(sa), len(sb)
    sp = np.sqrt(((n1 - 1) * sa.var(ddof=1) + (n2 - 1) * sb.var(ddof=1)) / (n1 + n2 - 2))
    d = (sb.mean() - sa.mean()) / sp if sp else np.nan
    abs_lift = sb.mean() - sa.mean()
    rel_lift = abs_lift / sa.mean() if sa.mean() else np.nan
    return {"mean_a": float(sa.mean()), "mean_b": float(sb.mean()), "n_a": n1, "n_b": n2,
            "t": float(t), "p": float(p), "cohens_d": float(d), "abs_lift": float(abs_lift),
            "rel_lift": float(rel_lift), "equal_var": bool(equal_var), "alpha": alpha,
            "labels": labels, "kind": "continuous"}


def interpret_design(res: dict) -> str:
    lines = ["### A/B test tasarımı — gereken örneklem"]
    mtxt = f"%{res['mde']*100:.0f} bağıl" if res["relative"] else f"{res['mde']:.4f} mutlak"
    lines.append(I.bullet(f"Taban dönüşüm %{res['p1']*100:.2f} → hedeflenen %{res['p2']*100:.2f} "
                          f"(saptanacak en küçük etki: {mtxt})."))
    lines.append(I.bullet(f"%{res['power']*100:.0f} güç ve α={res['alpha']:g} için: "
                          f"**kol başına {res['n_per_arm']:,} gözlem** (toplam {res['n_total']:,})."))
    if "days" in res:
        lines.append(I.bullet(f"Günlük ~{res['daily_traffic']:,} trafik ile testin süresi ≈ **{res['days']} gün**."))
    lines.append(I.bullet("Bu örneklemden önce durmak (peeking) yanlış pozitif riskini şişirir; hedef örnekleme ulaşın "
                          "ya da sıralı test düzeltmesi kullanın."))
    return I.joinlines(lines)


def interpret_analysis(res: dict) -> str:
    a = res["alpha"]
    lines = ["### A/B test sonucu"]
    if res["kind"] == "conversion":
        lines.append(I.bullet(f"Kontrol (A): %{res['p_a']*100:.2f} ({res['s_a']}/{res['n_a']}) · "
                              f"Varyant (B): %{res['p_b']*100:.2f} ({res['s_b']}/{res['n_b']})"))
        lines.append(I.bullet(f"Mutlak fark: {res['abs_lift']*100:+.2f} puan · Bağıl değişim: %{res['rel_lift']*100:+.1f}"))
        lines.append(I.bullet(f"z = {res['z']:.3f}, p = {res['p']:.4g} {I.stars(res['p'])}"))
        lines.append(I.bullet(f"Farkın %{int((1-a)*100)} güven aralığı: "
                              f"[{res['ci'][0]*100:+.2f}, {res['ci'][1]*100:+.2f}] puan"))
    else:
        l = res["labels"]
        lines.append(I.bullet(f"{l[0]} ortalama: {res['mean_a']:.4f} · {l[1]} ortalama: {res['mean_b']:.4f}"))
        lines.append(I.bullet(f"Mutlak fark: {res['abs_lift']:+.4f} · Bağıl: %{res['rel_lift']*100:+.1f} · "
                              f"Cohen's d = {res['cohens_d']:.3f} ({I.cohens_d_label(res['cohens_d'])})"))
        lines.append(I.bullet(f"t = {res['t']:.3f}, p = {res['p']:.4g} {I.stars(res['p'])}"))
    lines.append("")
    if res["p"] < a:
        winner = "B (varyant)" if res.get("abs_lift", 0) > 0 else "A (kontrol)"
        lines.append(I.bullet(f"**Karar:** Fark istatistiksel olarak anlamlı (p < {a:g}). **{winner}** daha iyi görünüyor. "
                              "Pratik anlamlılığı da (etki büyüklüğü/iş etkisi) değerlendirin."))
    else:
        lines.append(I.bullet(f"**Karar:** Fark anlamlı değil (p ≥ {a:g}). Varyantın üstün olduğuna dair yeterli kanıt yok; "
                              "örneklem yetersiz olabilir veya gerçek etki küçük olabilir."))
    return I.joinlines(lines)
