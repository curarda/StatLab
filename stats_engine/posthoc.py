"""İleri ANOVA ve post-hoc testleri: Welch ANOVA, Games-Howell, Dunn, Nemenyi, RM ANOVA."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from . import interpret as I


def manova(df: pd.DataFrame, dvs: list[str], group: str, alpha: float = 0.05) -> dict:
    """Çok değişkenli ANOVA — birden çok bağımlı değişkeni birlikte test eder."""
    from statsmodels.multivariate.manova import MANOVA
    cols = dvs + [group]
    data = df[cols].copy()
    for v in dvs:
        data[v] = pd.to_numeric(data[v], errors="coerce")
    data = data.dropna()
    if data[group].nunique() < 2 or len(dvs) < 2:
        return {"error": "MANOVA için ≥2 sayısal bağımlı değişken ve ≥2 gruplu bir faktör gerekir."}
    formula = " + ".join(f"Q('{v}')" for v in dvs) + f" ~ C(Q('{group}'))"
    try:
        mv = MANOVA.from_formula(formula, data=data).mv_test()
    except Exception as e:
        return {"error": f"MANOVA hesaplanamadı: {e}"}
    term = [k for k in mv.results if k != "Intercept"]
    key = term[0] if term else list(mv.results)[0]
    stat = mv.results[key]["stat"]
    def _get(name):
        row = stat.loc[name]
        return {"value": float(row["Value"]), "F": float(row["F Value"]), "p": float(row["Pr > F"])}
    return {"dvs": dvs, "group": group, "n": int(len(data)), "k": int(data[group].nunique()),
            "wilks": _get("Wilks' lambda"), "pillai": _get("Pillai's trace"),
            "hotelling": _get("Hotelling-Lawley trace"), "alpha": alpha}


def interpret_manova(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    w, p = res["wilks"], res["pillai"]
    lines = [f"### MANOVA — [{', '.join(res['dvs'])}] ~ {res['group']}"]
    lines.append(I.bullet(f"Wilks' λ = {w['value']:.4f}, F = {w['F']:.3f}, p = {w['p']:.4g} {I.stars(w['p'])}"))
    lines.append(I.bullet(f"Pillai's trace = {p['value']:.4f}, F = {p['F']:.3f}, p = {p['p']:.4g} "
                          "(gruplar/örneklem dengesizse Pillai daha güvenilir)"))
    lines.append("**Karar:** " + I.p_sentence(w["p"], a))
    if w["p"] < a:
        lines.append(I.bullet(f"Gruplar, bağımlı değişkenlerin **bileşimi** açısından anlamlı biçimde farklı. "
                              "Hangi değişken(ler)in sürüklediğini görmek için ardından tek tek ANOVA yapın "
                              "(ama MANOVA I. tip hatayı kontrol ederek genel farkı doğrular)."))
    else:
        lines.append(I.bullet("Değişkenlerin çok değişkenli bileşiminde gruplar arası anlamlı fark yok."))
    lines.append(I.bullet("MANOVA, ilişkili birden çok sonucu **birlikte** sınayarak ayrı ANOVA'ların "
                          "şişirdiği yanlış-pozitif riskini önler."))
    return I.joinlines(lines)


def welch_anova(df: pd.DataFrame, value: str, group: str, alpha: float = 0.05) -> dict:
    """Varyansların eşit olmadığı durumda tek yönlü ANOVA (Welch)."""
    import pingouin as pg
    data = df[[value, group]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pg.welch_anova(data, dv=value, between=group)
    r = res.iloc[0]
    return {"test": "Welch ANOVA (eşit olmayan varyans)", "F": float(r["F"]),
            "df1": float(r["ddof1"]), "df2": float(r["ddof2"]), "p": float(r["p_unc"]),
            "eta2": float(r["np2"]), "alpha": alpha, "value": value, "group": group}


def games_howell(df: pd.DataFrame, value: str, group: str, alpha: float = 0.05) -> dict:
    """Games-Howell post-hoc (varyanslar eşit değilken ANOVA sonrası ikili karşılaştırma)."""
    import pingouin as pg
    data = df[[value, group]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gh = pg.pairwise_gameshowell(data, dv=value, between=group)
    tbl = pd.DataFrame({
        "Grup 1": gh["A"], "Grup 2": gh["B"], "Ort. fark": gh["diff"],
        "t": gh["T"], "sd": gh["df"], "p": gh["pval"],
        "Anlamlı?": ["Evet" if p < alpha else "Hayır" for p in gh["pval"]],
    })
    return {"test": "Games-Howell post-hoc", "table": tbl, "alpha": alpha}


def dunn(df: pd.DataFrame, value: str, group: str, adjust: str = "holm", alpha: float = 0.05) -> dict:
    """Dunn post-hoc (Kruskal-Wallis sonrası, parametrik olmayan ikili karşılaştırma)."""
    import scikit_posthocs as sp
    data = df[[value, group]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna()
    m = sp.posthoc_dunn(data, val_col=value, group_col=group, p_adjust=adjust)
    return {"test": f"Dunn post-hoc ({adjust} düzeltmeli)", "matrix": m, "alpha": alpha,
            "pairs": _pairs_from_matrix(m, alpha)}


def nemenyi_friedman(df_wide: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Nemenyi post-hoc (Friedman sonrası). df_wide: satır=denek, sütun=koşul."""
    import scikit_posthocs as sp
    m = sp.posthoc_nemenyi_friedman(df_wide.values)
    m.index = df_wide.columns
    m.columns = df_wide.columns
    return {"test": "Nemenyi post-hoc (Friedman sonrası)", "matrix": m, "alpha": alpha,
            "pairs": _pairs_from_matrix(m, alpha)}


def rm_anova(df_long: pd.DataFrame, dv: str, within: str, subject: str, alpha: float = 0.05) -> dict:
    """Tekrarlı ölçüm ANOVA (within-subjects). Uzun formatta veri gerekir."""
    import pingouin as pg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pg.rm_anova(df_long, dv=dv, within=within, subject=subject)
    r = res.iloc[0]
    sph = None
    try:
        sp_res = pg.sphericity(df_long, dv=dv, within=within, subject=subject)
        sph = float(sp_res[-1])  # pval
    except Exception:
        pass
    return {"test": "Tekrarlı ölçüm ANOVA", "F": float(r["F"]), "df1": float(r["ddof1"]),
            "df2": float(r["ddof2"]), "p": float(r["p_unc"]), "eta2": float(r.get("ng2", np.nan)),
            "sphericity_p": sph, "alpha": alpha}


def rm_anova_from_wide(df: pd.DataFrame, cols: list[str], alpha: float = 0.05) -> dict:
    """Geniş formattan (her sütun bir koşul) RM ANOVA."""
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if len(data) < 2:
        return {"error": "Yeterli tam gözlem yok."}
    long = data.reset_index().melt(id_vars="index", value_vars=cols, var_name="kosul", value_name="deger")
    long = long.rename(columns={"index": "denek"})
    return rm_anova(long, "deger", "kosul", "denek", alpha)


def _pairs_from_matrix(m: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    cols = list(m.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            p = m.iloc[i, j]
            rows.append({"Grup 1": cols[i], "Grup 2": cols[j], "p": float(p),
                         "Anlamlı?": "Evet" if p < alpha else "Hayır"})
    return pd.DataFrame(rows)


# ---------------- yorumlar ----------------

def interpret_welch(res: dict) -> str:
    a = res["alpha"]
    lines = [f"### {res['test']} — {res['value']} ~ {res['group']}"]
    lines.append(I.bullet(f"F({res['df1']:.0f}, {res['df2']:.1f}) = {res['F']:.3f}, p = {res['p']:.4g} "
                          f"{I.stars(res['p'])}, η² = {res['eta2']:.3f}"))
    lines.append("**Karar:** " + I.p_sentence(res["p"], a))
    lines.append(I.bullet("Welch ANOVA varyans homojenliği varsaymaz; Levene anlamlıysa standart ANOVA yerine bunu kullanın. "
                          "Anlamlıysa Games-Howell post-hoc ile ikilileri inceleyin."))
    return I.joinlines(lines)


def interpret_posthoc(res: dict) -> str:
    lines = [f"### {res['test']}"]
    tbl = res.get("table")
    if tbl is not None:
        sig = tbl[tbl["Anlamlı?"] == "Evet"]
    else:
        sig = res["pairs"][res["pairs"]["Anlamlı?"] == "Evet"]
    if sig.empty:
        lines.append(I.bullet("Hiçbir ikili karşılaştırma anlamlı değil."))
    else:
        lines.append(I.bullet(f"**{len(sig)} ikili** anlamlı biçimde farklı:"))
        for _, r in sig.iterrows():
            g1 = r.get("Grup 1"); g2 = r.get("Grup 2")
            lines.append(I.bullet(f"{g1} ↔ {g2}: p = {r['p']:.4g}"))
    return I.joinlines(lines)


def interpret_rm(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    lines = [f"### {res['test']}"]
    lines.append(I.bullet(f"F({res['df1']:.0f}, {res['df2']:.0f}) = {res['F']:.3f}, p = {res['p']:.4g} "
                          f"{I.stars(res['p'])}, η²(genelleştirilmiş) = {res['eta2']:.3f}"))
    if res.get("sphericity_p") is not None:
        ok = res["sphericity_p"] >= a
        lines.append(I.bullet(f"Küresellik (Mauchly) p = {res['sphericity_p']:.4g} → "
                              + ("varsayım korunuyor." if ok else "**ihlal**; Greenhouse-Geisser düzeltmesi önerilir.")))
    lines.append("**Karar:** " + I.p_sentence(res["p"], a))
    lines.append(I.bullet("Aynı deneklerin 3+ koşulda tekrar ölçüldüğü tasarımlar içindir; "
                          "anlamlıysa Nemenyi/eşleştirilmiş post-hoc ile koşulları karşılaştırın."))
    return I.joinlines(lines)
