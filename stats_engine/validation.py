"""Örneklem-dışı doğrulama: k-kat çapraz doğrulama ve eğitim/test bölmesi.

Ayrım: metrikler (ham) ile aşırı-uyum yorumu (yorum katmanı) ayrı döner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import interpret as I


def _prep(df, y, x_vars):
    data = df[[y] + x_vars].apply(pd.to_numeric, errors="coerce").dropna()
    return data[x_vars].values, data[y].values, data


def cv_regression(df: pd.DataFrame, y: str, x_vars: list[str], folds: int = 5) -> dict:
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    X, yv, data = _prep(df, y, x_vars)
    if len(data) < folds + 2:
        return {"error": "Çapraz doğrulama için yeterli gözlem yok."}
    model = LinearRegression()
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    # in-sample
    model.fit(X, yv)
    r2_in = r2_score(yv, model.predict(X))
    # out-of-sample (CV)
    pred = cross_val_predict(model, X, yv, cv=kf)
    r2_cv = r2_score(yv, pred)
    rmse_cv = float(np.sqrt(mean_squared_error(yv, pred)))
    # kat başına
    fold_r2 = []
    for tr, te in kf.split(X):
        model.fit(X[tr], yv[tr])
        fold_r2.append(r2_score(yv[te], model.predict(X[te])))
    return {"task": "regresyon", "y": y, "x_vars": x_vars, "folds": folds, "n": len(data),
            "r2_in": float(r2_in), "r2_cv": float(r2_cv), "rmse_cv": rmse_cv,
            "gap": float(r2_in - r2_cv), "fold_scores": [float(s) for s in fold_r2],
            "pred": pred, "actual": yv}


def cv_logistic(df: pd.DataFrame, y: str, x_vars: list[str], positive, folds: int = 5) -> dict:
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    data = df[[y] + x_vars].copy()
    for v in x_vars:
        data[v] = pd.to_numeric(data[v], errors="coerce")
    data = data.dropna()
    yb = (data[y].astype(str) == str(positive)).astype(int).values
    if yb.sum() < folds or (len(yb) - yb.sum()) < folds:
        return {"error": "Her sınıfta kat sayısı kadar gözlem gerekir."}
    X = data[x_vars].values
    model = LogisticRegression(max_iter=1000)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    model.fit(X, yb)
    auc_in = roc_auc_score(yb, model.predict_proba(X)[:, 1])
    proba = cross_val_predict(model, X, yb, cv=skf, method="predict_proba")[:, 1]
    auc_cv = roc_auc_score(yb, proba)
    acc_cv = accuracy_score(yb, (proba >= 0.5).astype(int))
    return {"task": "lojistik", "y": y, "x_vars": x_vars, "positive": positive, "folds": folds,
            "n": len(data), "auc_in": float(auc_in), "auc_cv": float(auc_cv), "acc_cv": float(acc_cv),
            "gap": float(auc_in - auc_cv), "proba": proba, "ytrue": yb}


def holdout(df: pd.DataFrame, y: str, x_vars: list[str], task: str = "regresyon",
            test_size: float = 0.25, positive=None) -> dict:
    from sklearn.model_selection import train_test_split
    if task == "lojistik":
        data = df[[y] + x_vars].copy()
        for v in x_vars:
            data[v] = pd.to_numeric(data[v], errors="coerce")
        data = data.dropna()
        yb = (data[y].astype(str) == str(positive)).astype(int).values
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        Xtr, Xte, ytr, yte = train_test_split(data[x_vars].values, yb, test_size=test_size,
                                              random_state=42, stratify=yb)
        m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        return {"task": task, "train_auc": float(roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1])),
                "test_auc": float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1])),
                "n_train": len(ytr), "n_test": len(yte)}
    X, yv, data = _prep(df, y, x_vars)
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    Xtr, Xte, ytr, yte = train_test_split(X, yv, test_size=test_size, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    return {"task": task, "train_r2": float(r2_score(ytr, m.predict(Xtr))),
            "test_r2": float(r2_score(yte, m.predict(Xte))), "n_train": len(ytr), "n_test": len(yte)}


def interpret_cv(res: dict) -> str:
    if "error" in res:
        return res["error"]
    lines = [f"### {res['task'].capitalize()} — örneklem-dışı doğrulama (yorum katmanı)"]
    if res["task"] == "regresyon":
        lines.append(I.bullet(f"Örneklem-içi R² = {res['r2_in']:.3f} → **çapraz doğrulama R² = {res['r2_cv']:.3f}** "
                              f"(RMSE = {res['rmse_cv']:.3f}, {res['folds']} kat)."))
        metric, gap = "R²", res["gap"]
    else:
        lines.append(I.bullet(f"Örneklem-içi AUC = {res['auc_in']:.3f} → **çapraz doğrulama AUC = {res['auc_cv']:.3f}** "
                              f"(doğruluk = {res['acc_cv']:.3f}, {res['folds']} kat)."))
        metric, gap = "AUC", res["gap"]
    if gap > 0.15:
        lines.append(I.bullet(f"⚠️ {metric} örneklem-dışında **{gap:.2f} düştü** → belirgin **aşırı uyum (overfitting)**. "
                              "Model gördüğü veriyi ezberlemiş; yeni veride bu kadar iyi olmayacak. "
                              "Değişken azalt / düzenlileştirme (ridge/lasso) düşün."))
    elif gap > 0.05:
        lines.append(I.bullet(f"{metric} örneklem-dışında {gap:.2f} düştü → hafif iyimserlik, kabul edilebilir."))
    else:
        lines.append(I.bullet(f"Örneklem-içi ve dışı {metric} çok yakın → model **iyi genelleşiyor**, güvenilir. ✅"))
    lines.append(I.bullet("Örneklem-dışı metrik, modelin **görmediği veride** gerçekte ne kadar iyi olacağının "
                          "dürüst tahminidir — raporlanması gereken sayı budur."))
    return I.joinlines(lines)


def interpret_holdout(res: dict) -> str:
    m = "R²" if res["task"] == "regresyon" else "AUC"
    tr = res.get("train_r2", res.get("train_auc"))
    te = res.get("test_r2", res.get("test_auc"))
    lines = [f"### Eğitim/Test bölmesi ({res['n_train']}/{res['n_test']})"]
    lines.append(I.bullet(f"Eğitim {m} = {tr:.3f} → **Test {m} = {te:.3f}**."))
    gap = tr - te
    lines.append(I.bullet("Aşırı uyum riski düşük ✅" if gap < 0.05 else
                          "Hafif aşırı uyum." if gap < 0.15 else "⚠️ Belirgin aşırı uyum — test performansı esastır."))
    return I.joinlines(lines)
