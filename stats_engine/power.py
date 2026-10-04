"""Güç analizi ve örneklem büyüklüğü hesabı."""
from __future__ import annotations

import numpy as np
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from . import interpret as I


def solve_ttest(effect: float, n: int | None, power: float | None, alpha: float = 0.05,
                ratio: float = 1.0, alternative: str = "two-sided") -> dict:
    """İki bağımsız grup t-testi. Verilmeyen bir tanesini (n veya power) çöz."""
    analysis = TTestIndPower()
    out = {"effect": effect, "alpha": alpha, "test": "İki örneklem t-testi (Cohen's d)"}
    if power is None and n is not None:
        p = analysis.power(effect_size=effect, nobs1=n, alpha=alpha, ratio=ratio, alternative=alternative)
        out.update(n1=n, power=float(p), solve_for="power")
    elif n is None and power is not None:
        nn = analysis.solve_power(effect_size=effect, power=power, alpha=alpha, ratio=ratio, alternative=alternative)
        out.update(power=power, n1=int(np.ceil(nn)), solve_for="n")
    else:
        return {"error": "n veya güç değerlerinden yalnızca birini boş bırakın."}
    return out


def solve_anova(effect: float, n_groups: int, n: int | None, power: float | None, alpha: float = 0.05) -> dict:
    """Tek yönlü ANOVA (Cohen's f). n = grup başına gözlem."""
    analysis = FTestAnovaPower()
    out = {"effect": effect, "alpha": alpha, "k": n_groups, "test": "Tek yönlü ANOVA (Cohen's f)"}
    if power is None and n is not None:
        total = n * n_groups
        p = analysis.power(effect_size=effect, nobs=total, alpha=alpha, k_groups=n_groups)
        out.update(n_per_group=n, power=float(p), solve_for="power")
    elif n is None and power is not None:
        total = analysis.solve_power(effect_size=effect, power=power, alpha=alpha, k_groups=n_groups)
        out.update(power=power, n_per_group=int(np.ceil(total / n_groups)), solve_for="n")
    else:
        return {"error": "n veya güç değerlerinden yalnızca birini boş bırakın."}
    return out


def solve_proportions(p1: float, p2: float, n: int | None, power: float | None, alpha: float = 0.05) -> dict:
    """İki oran karşılaştırması."""
    effect = proportion_effectsize(p1, p2)
    analysis = NormalIndPower()
    out = {"effect": float(effect), "alpha": alpha, "p1": p1, "p2": p2, "test": "İki oran testi"}
    if power is None and n is not None:
        p = analysis.power(effect_size=abs(effect), nobs1=n, alpha=alpha)
        out.update(n1=n, power=float(p), solve_for="power")
    elif n is None and power is not None:
        nn = analysis.solve_power(effect_size=abs(effect), power=power, alpha=alpha)
        out.update(power=power, n1=int(np.ceil(nn)), solve_for="n")
    else:
        return {"error": "n veya güç değerlerinden yalnızca birini boş bırakın."}
    return out


def power_curve(kind: str, effect: float, alpha: float, n_range, n_groups: int = 3) -> dict:
    """Örneklem büyüklüğüne karşı güç eğrisi (grafik için)."""
    powers = []
    for n in n_range:
        if kind == "ttest":
            powers.append(TTestIndPower().power(effect_size=effect, nobs1=int(n), alpha=alpha))
        elif kind == "anova":
            powers.append(FTestAnovaPower().power(effect_size=effect, nobs=int(n) * n_groups,
                                                  alpha=alpha, k_groups=n_groups))
        else:
            powers.append(NormalIndPower().power(effect_size=effect, nobs1=int(n), alpha=alpha))
    return {"n": list(n_range), "power": powers}


def interpret_power(res: dict) -> str:
    if "error" in res:
        return res["error"]
    lines = [f"### {res['test']} — güç analizi"]
    lines.append(I.bullet(f"Etki büyüklüğü = {res['effect']:.3f}, α = {res['alpha']:g}."))
    if res["solve_for"] == "n":
        nlabel = "grup başına" if "n_per_group" in res else "1. grupta"
        nval = res.get("n_per_group", res.get("n1"))
        lines.append(I.bullet(f"**%{res['power']*100:.0f} güç** için gereken örneklem: **{nlabel} {nval}** gözlem."))
        lines.append(I.bullet("Yeterli örneklem, gerçek bir etkiyi kaçırma (II. tip hata) riskini azaltır."))
    else:
        p = res["power"]
        nval = res.get("n_per_group", res.get("n1"))
        q = ("yetersiz (<0.80)" if p < 0.80 else "yeterli (≥0.80)" if p < 0.95 else "çok yüksek")
        lines.append(I.bullet(f"n = {nval} ile elde edilen **istatistiksel güç = {p:.3f}** → {q}."))
        if p < 0.80:
            lines.append(I.bullet("Güç 0.80'in altında: örneklemi artırmayı düşünün; aksi halde gerçek etki gözden kaçabilir."))
    lines.append(I.bullet("Genel kabul: hedef güç ≥ 0.80 (II. tip hata ≤ %20)."))
    return I.joinlines(lines)
