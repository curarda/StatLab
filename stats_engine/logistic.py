"""İkili (binary) lojistik regresyon + sınıflandırma metrikleri + ROC."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import interpret as I


def fit_logit(df: pd.DataFrame, y: str, x_vars: list[str], positive_label,
              categorical: list[str] | None = None, alpha: float = 0.05,
              threshold: float = 0.5) -> dict:
    """İkili lojistik regresyon. y, positive_label'e eşitse 1, değilse 0 kodlanır."""
    categorical = categorical or []
    data = df[[y] + x_vars].copy()
    for v in [v for v in x_vars if v not in categorical]:
        data[v] = pd.to_numeric(data[v], errors="coerce")
    data = data.dropna()
    if data.empty:
        return {"error": "Geçerli veri yok."}

    # İkili hedef
    ybin = (data[y].astype(str) == str(positive_label)).astype(int)
    if ybin.nunique() < 2:
        return {"error": f"Hedef '{y}' seçilen pozitif sınıfa göre tek düzeyli; iki sınıf gerekir."}
    data = data.assign(__target__=ybin.values)

    terms = [f"C(Q('{v}'))" if v in categorical else f"Q('{v}')" for v in x_vars]
    formula = "__target__ ~ " + " + ".join(terms)
    try:
        model = smf.logit(formula, data=data).fit(disp=False, maxiter=200)
    except Exception as e:
        return {"error": f"Model uyduramadı (olası tam ayrışma/perfect separation): {e}"}

    params = model.params
    conf = model.conf_int(alpha=alpha)
    rows = []
    for name in params.index:
        b = float(params[name])
        rows.append({
            "Terim": _pretty(name),
            "Katsayı (β)": b,
            "Odds Oranı": float(np.exp(b)),
            "OR CI alt": float(np.exp(conf.loc[name, 0])),
            "OR CI üst": float(np.exp(conf.loc[name, 1])),
            "z": float(model.tvalues[name]),
            "p": float(model.pvalues[name]),
            "Anlam": I.stars(float(model.pvalues[name])) or "—",
        })
    coef_df = pd.DataFrame(rows)

    # Tahmin ve sınıflandırma
    probs = np.asarray(model.predict(data))
    ytrue = data["__target__"].values
    pred = (probs >= threshold).astype(int)
    cm = _confusion(ytrue, pred)
    metrics = _class_metrics(cm)
    roc = _roc(ytrue, probs)

    return {
        "y": y, "positive_label": positive_label, "x_vars": x_vars,
        "n": int(model.nobs), "coef_df": coef_df, "coef_params": params,
        "pseudo_r2": float(model.prsquared), "llr_p": float(model.llr_pvalue),
        "aic": float(model.aic), "bic": float(model.bic),
        "confusion": cm, "metrics": metrics, "roc": roc,
        "threshold": threshold, "alpha": alpha, "probs": probs, "ytrue": ytrue,
        "summary_text": model.summary().as_text(),
    }


def _pretty(name: str) -> str:
    return (name.replace("Q('", "").replace("')", "").replace("C(", "")
                .replace(")", "").replace("[T.", " = ").replace("]", ""))


def _confusion(ytrue, pred) -> dict:
    tp = int(np.sum((pred == 1) & (ytrue == 1)))
    tn = int(np.sum((pred == 0) & (ytrue == 0)))
    fp = int(np.sum((pred == 1) & (ytrue == 0)))
    fn = int(np.sum((pred == 0) & (ytrue == 1)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def _class_metrics(cm: dict) -> dict:
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec and (prec + rec)) else np.nan
    return {"accuracy": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1}


def _roc(ytrue, probs):
    order = np.argsort(-probs)
    yt = np.asarray(ytrue)[order]
    P = yt.sum()
    N = len(yt) - P
    if P == 0 or N == 0:
        return {"fpr": [0, 1], "tpr": [0, 1], "auc": np.nan}
    tpr, fpr = [0.0], [0.0]
    tp = fp = 0
    for label in yt:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / P)
        fpr.append(fp / N)
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trap(tpr, fpr))
    return {"fpr": fpr, "tpr": tpr, "auc": auc}


def predict_proba(res: dict, values: dict) -> float:
    """Verilen sayısal öngörücü değerleri için tahmini olasılık (yalnızca sayısal terimler)."""
    params = res["coef_params"]
    z = float(params.get("Intercept", 0.0))
    for name, coef in params.items():
        if name == "Intercept":
            continue
        var = _pretty(name)
        if var in values:
            z += float(coef) * float(values[var])
    return float(1 / (1 + np.exp(-z)))


def fit_multinomial(df: pd.DataFrame, y: str, x_vars: list[str],
                    categorical: list[str] | None = None, alpha: float = 0.05) -> dict:
    """Çok sınıflı (multinomial) lojistik regresyon — hedefte 3+ kategori."""
    categorical = categorical or []
    data = df[[y] + x_vars].copy()
    for v in [v for v in x_vars if v not in categorical]:
        data[v] = pd.to_numeric(data[v], errors="coerce")
    data = data.dropna()
    classes = sorted(data[y].astype(str).unique().tolist())
    if len(classes) < 3:
        return {"error": "Multinomial için hedefte en az 3 sınıf gerekir (2 sınıf → İkili modu kullanın)."}
    code_map = {c: i for i, c in enumerate(classes)}
    data = data.assign(__t__=data[y].astype(str).map(code_map).values)

    terms = [f"C(Q('{v}'))" if v in categorical else f"Q('{v}')" for v in x_vars]
    formula = "__t__ ~ " + " + ".join(terms)
    try:
        model = smf.mnlogit(formula, data=data).fit(disp=False, maxiter=300)
    except Exception as e:
        return {"error": f"Model uyduramadı: {e}"}

    params = model.params      # index=öngörücü, sütun=sınıf (baseline hariç)
    pvals = model.pvalues
    # Sınıf başına katsayı tabloları
    per_class = {}
    for j, col in enumerate(params.columns):
        cls = classes[j + 1]   # baseline = classes[0]
        rows = []
        for name in params.index:
            b = float(params.loc[name, col])
            rows.append({"Terim": _pretty(name), "Katsayı (β)": b, "Odds Oranı": float(np.exp(b)),
                         "p": float(pvals.loc[name, col]), "Anlam": I.stars(float(pvals.loc[name, col])) or "—"})
        per_class[cls] = pd.DataFrame(rows)

    probs = np.asarray(model.predict(data))
    pred_code = probs.argmax(axis=1)
    ytrue = data["__t__"].values
    acc = float(np.mean(pred_code == ytrue))
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(ytrue, pred_code):
        cm[int(t), int(p)] += 1
    cm_df = pd.DataFrame(cm, index=[f"Gerçek {c}" for c in classes],
                         columns=[f"Tahmin {c}" for c in classes])
    return {"y": y, "x_vars": x_vars, "classes": classes, "baseline": classes[0],
            "n": int(model.nobs), "pseudo_r2": float(model.prsquared),
            "llr_p": float(model.llr_pvalue), "aic": float(model.aic), "bic": float(model.bic),
            "per_class": per_class, "accuracy": acc, "confusion": cm_df,
            "alpha": alpha, "multinomial": True, "summary_text": model.summary().as_text()}


def fit_ordinal(df: pd.DataFrame, y: str, x_vars: list[str], order: list,
                alpha: float = 0.05) -> dict:
    """Sıralı (ordinal) lojistik regresyon — hedef sıralı kategorik (ör. düşük<orta<yüksek)."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    data = df[[y] + x_vars].copy()
    for v in x_vars:
        data[v] = pd.to_numeric(data[v], errors="coerce")
    data = data.dropna()
    cat = pd.Categorical(data[y].astype(str), categories=[str(o) for o in order], ordered=True)
    if cat.isna().any():
        return {"error": "Hedefteki bazı değerler verilen sıralamada yok. Sıralamayı kontrol edin."}
    data = data.assign(__t__=cat)
    try:
        model = OrderedModel(data["__t__"], data[x_vars], distr="logit").fit(method="bfgs", disp=False)
    except Exception as e:
        return {"error": f"Model uyduramadı: {e}"}
    params = model.params
    conf = model.conf_int(alpha=alpha)
    rows = []
    for name in x_vars:  # yalnızca eğim katsayıları (eşikler hariç)
        b = float(params[name])
        rows.append({"Terim": name, "Katsayı (β)": b, "Odds Oranı": float(np.exp(b)),
                     "OR CI alt": float(np.exp(conf.loc[name, 0])), "OR CI üst": float(np.exp(conf.loc[name, 1])),
                     "z": float(model.tvalues[name]), "p": float(model.pvalues[name]),
                     "Anlam": I.stars(float(model.pvalues[name])) or "—"})
    return {"y": y, "x_vars": x_vars, "order": [str(o) for o in order], "n": int(model.nobs),
            "coef_df": pd.DataFrame(rows), "pseudo_r2": float(model.prsquared),
            "aic": float(model.aic), "alpha": alpha, "ordinal": True,
            "summary_text": model.summary().as_text()}


def interpret_ordinal(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    lines = [f"### Sıralı (ordinal) lojistik regresyon — hedef: **{res['y']}**"]
    lines.append(I.bullet(f"Sıralama: {' < '.join(map(str, res['order']))}. "
                          f"n = {res['n']}, sözde-R² = {res['pseudo_r2']:.3f}, AIC = {res['aic']:.1f}."))
    lines.append(I.bullet("Orantılı olasılıklar (proportional odds) varsayımı: her öngörücünün etkisi "
                          "tüm eşiklerde aynı kabul edilir."))
    lines.append("\n**Katsayılar (kümülatif odds oranları)**")
    for _, r in res["coef_df"].iterrows():
        orr, p = r["Odds Oranı"], r["p"]
        if orr >= 1:
            eff = f"daha yüksek kategoride olma odds'unu **{orr:.3f} kat artırır**"
        else:
            eff = f"daha yüksek kategoride olma odds'unu **{orr:.3f} kata düşürür**"
        sig = "**anlamlı**" if p < a else "anlamlı değil"
        lines.append(I.bullet(f"**{r['Terim']}**: OR = {orr:.3f} {r['Anlam']} (p={p:.4g}) → 1 birim artış {eff}. Etki {sig}."))
    return I.joinlines(lines)


def interpret_multinomial(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    lines = [f"### Çok sınıflı (multinomial) lojistik regresyon — hedef: **{res['y']}**"]
    lines.append(I.bullet(f"{len(res['classes'])} sınıf: {', '.join(map(str, res['classes']))}. "
                          f"**Referans (baseline) sınıf: {res['baseline']}** — tüm karşılaştırmalar buna göredir."))
    lines.append(I.bullet(f"n = {res['n']}, McFadden sözde-R² = {res['pseudo_r2']:.3f}, "
                          f"AIC = {res['aic']:.1f}, doğruluk = {res['accuracy']:.3f}."))
    lines.append(I.bullet("LLR testi p = %.4g → model %s." % (
        res["llr_p"], "anlamlı" if res["llr_p"] < a else "anlamlı değil")))
    for cls, tbl in res["per_class"].items():
        lines.append(f"\n**{res['baseline']} → {cls} geçişi (bu sınıfta olma olasılığını etkileyenler):**")
        sig = tbl[(tbl["Terim"].str.lower() != "intercept") & (tbl["p"] < a)]
        if sig.empty:
            lines.append(I.bullet("Bu sınıf için anlamlı öngörücü yok."))
        for _, r in sig.iterrows():
            orr = r["Odds Oranı"]
            eff = (f"olasılığı {orr:.2f} kat **artırır**" if orr > 1 else f"olasılığı {orr:.2f} kata **düşürür**")
            lines.append(I.bullet(f"{r['Terim']}: OR = {orr:.3f} (p={r['p']:.4g}) → 1 birim artış, {res['baseline']} yerine "
                                  f"{cls} sınıfında olma {eff}."))
    return I.joinlines(lines)


def interpret_logit(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    m = res["metrics"]
    auc = res["roc"]["auc"]
    lines = [f"### Lojistik regresyon — hedef: **{res['y']} = {res['positive_label']}** (pozitif sınıf)"]
    lines.append(I.bullet(f"n = {res['n']}, McFadden sözde-R² = {res['pseudo_r2']:.3f}, "
                          f"AIC = {res['aic']:.1f}, BIC = {res['bic']:.1f}"))

    lines.append("\n**1) Modelin genel anlamlılığı (LLR testi)**")
    lines.append(I.bullet(f"Olabilirlik oranı testi p = {res['llr_p']:.4g} → "
                          + ("model bir bütün olarak **anlamlı**; öngörücüler sonucu açıklıyor."
                             if res['llr_p'] < a else "model bir bütün olarak **anlamlı değil**.")))

    lines.append("\n**2) Katsayılar ve odds oranları (OR)**")
    for _, row in res["coef_df"].iterrows():
        term, b, orr, p = row["Terim"], row["Katsayı (β)"], row["Odds Oranı"], row["p"]
        if term.lower() == "intercept":
            lines.append(I.bullet(f"**Sabit**: β = {b:.4f} (temel log-odds)."))
            continue
        if orr >= 1:
            pct = (orr - 1) * 100
            eff = f"odds'u **{orr:.3f} katına** çıkarır (~%{pct:.1f} artış)"
        else:
            pct = (1 - orr) * 100
            eff = f"odds'u **{orr:.3f} katına** düşürür (~%{pct:.1f} azalış)"
        sig = "**anlamlı**" if p < a else "anlamlı değil"
        lines.append(I.bullet(f"**{term}**: OR = {orr:.3f} {row['Anlam']} (p = {p:.4g}). "
                              f"Diğerleri sabitken 1 birim artış {eff}. Bu etki {sig}."))

    lines.append("\n**3) Sınıflandırma başarısı (eşik = %.2f)**" % res["threshold"])
    lines.append(I.bullet(f"Doğruluk (accuracy) = {m['accuracy']:.3f} | Kesinlik (precision) = {m['precision']:.3f} | "
                          f"Duyarlılık (recall) = {m['recall']:.3f} | Özgüllük = {m['specificity']:.3f} | F1 = {m['f1']:.3f}"))
    if auc == auc:
        auc_q = ("ayırt etme gücü zayıf (~rastgele)" if auc < 0.6 else
                 "orta" if auc < 0.7 else "kabul edilebilir" if auc < 0.8 else
                 "iyi" if auc < 0.9 else "çok iyi/mükemmel")
        lines.append(I.bullet(f"ROC eğrisi altındaki alan AUC = {auc:.3f} → {auc_q}."))
    cm = res["confusion"]
    lines.append(I.bullet(f"Karışıklık matrisi — Doğru pozitif={cm['TP']}, Doğru negatif={cm['TN']}, "
                          f"Yanlış pozitif={cm['FP']}, Yanlış negatif={cm['FN']}."))
    return I.joinlines(lines)
