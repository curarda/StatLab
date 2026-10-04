"""Kümeleme (k-means, hiyerarşik) ve boyut indirgeme (PCA)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import interpret as I


def _prep(df: pd.DataFrame, cols: list[str], scale: bool = True):
    from sklearn.preprocessing import StandardScaler
    X = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    idx = X.index
    Xv = StandardScaler().fit_transform(X.values) if scale else X.values
    return Xv, X, idx


def kmeans(df: pd.DataFrame, cols: list[str], k: int, scale: bool = True) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    Xv, X, idx = _prep(df, cols, scale)
    if len(X) < k:
        return {"error": "Küme sayısı gözlemden fazla olamaz."}
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xv)
    labels = km.labels_
    sil = float(silhouette_score(Xv, labels)) if k > 1 and len(set(labels)) > 1 else np.nan
    centers = pd.DataFrame(
        df.loc[idx, cols].assign(__c=labels).groupby("__c")[cols].mean())
    centers.index.name = "Küme"
    sizes = pd.Series(labels).value_counts().sort_index()
    return {"labels": labels, "index": idx, "k": k, "inertia": float(km.inertia_),
            "silhouette": sil, "centers": centers.reset_index(),
            "sizes": sizes.to_dict(), "cols": cols, "X": X, "method": "k-means"}


def hierarchical(df: pd.DataFrame, cols: list[str], k: int, linkage: str = "ward", scale: bool = True) -> dict:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    Xv, X, idx = _prep(df, cols, scale)
    if len(X) < k:
        return {"error": "Küme sayısı gözlemden fazla olamaz."}
    hc = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(Xv)
    labels = hc.labels_
    sil = float(silhouette_score(Xv, labels)) if k > 1 and len(set(labels)) > 1 else np.nan
    centers = pd.DataFrame(df.loc[idx, cols].assign(__c=labels).groupby("__c")[cols].mean())
    centers.index.name = "Küme"
    sizes = pd.Series(labels).value_counts().sort_index()
    return {"labels": labels, "index": idx, "k": k, "silhouette": sil,
            "centers": centers.reset_index(), "sizes": sizes.to_dict(),
            "cols": cols, "X": X, "method": f"hiyerarşik ({linkage})", "Xv": Xv}


def elbow(df: pd.DataFrame, cols: list[str], k_max: int = 8, scale: bool = True) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    Xv, X, idx = _prep(df, cols, scale)
    k_max = min(k_max, len(X) - 1)
    ks, inertias, sils = [], [], []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xv)
        ks.append(k); inertias.append(float(km.inertia_))
        sils.append(float(silhouette_score(Xv, km.labels_)))
    best_k = ks[int(np.argmax(sils))] if sils else 2
    return {"k": ks, "inertia": inertias, "silhouette": sils, "best_k": best_k}


def pca(df: pd.DataFrame, cols: list[str], n_components: int | None = None, scale: bool = True) -> dict:
    from sklearn.decomposition import PCA
    Xv, X, idx = _prep(df, cols, scale)
    n_components = n_components or min(len(cols), len(X))
    p = PCA(n_components=n_components).fit(Xv)
    scores = p.transform(Xv)
    evr = p.explained_variance_ratio_
    loadings = pd.DataFrame(p.components_.T, index=cols,
                            columns=[f"PC{i+1}" for i in range(n_components)])
    var_df = pd.DataFrame({"Bileşen": [f"PC{i+1}" for i in range(n_components)],
                           "Açıklanan Varyans %": evr * 100,
                           "Kümülatif %": np.cumsum(evr) * 100})
    return {"explained": evr, "cum": np.cumsum(evr), "loadings": loadings,
            "var_df": var_df, "scores": scores, "cols": cols, "index": idx,
            "n_components": n_components}


def interpret_clustering(res: dict) -> str:
    if "error" in res:
        return res["error"]
    lines = [f"### {res['method'].capitalize()} kümeleme — {res['k']} küme"]
    sizes = ", ".join(f"Küme {c}: {n}" for c, n in res["sizes"].items())
    lines.append(I.bullet(f"Küme büyüklükleri: {sizes}."))
    s = res.get("silhouette")
    if s == s:
        q = ("zayıf/örtüşen kümeler (<0.25)" if s < 0.25 else "orta yapı (0.25–0.5)" if s < 0.5
             else "belirgin yapı (0.5–0.7)" if s < 0.7 else "çok güçlü ayrışma (>0.7)")
        lines.append(I.bullet(f"Silhouette skoru = {s:.3f} → {q}."))
    if "inertia" in res:
        lines.append(I.bullet(f"Küme içi kareler toplamı (inertia) = {res['inertia']:.1f} (düşük = daha sıkı kümeler)."))
    lines.append(I.bullet("Aşağıdaki küme merkezleri tablosu, her kümenin değişken ortalamalarını gösterir — "
                          "kümeleri bu profillere göre yorumlayın (ör. yüksek X + düşük Y grubu)."))
    return I.joinlines(lines)


def interpret_pca(res: dict, alpha_var: float = 0.80) -> str:
    lines = ["### Temel Bileşenler Analizi (PCA)"]
    cum = res["cum"]
    n80 = int(np.argmax(cum >= alpha_var) + 1) if np.any(cum >= alpha_var) else len(cum)
    lines.append(I.bullet(f"İlk bileşen varyansın %{res['explained'][0]*100:.1f}'ini açıklıyor. "
                          f"Varyansın %{alpha_var*100:.0f}'ine ulaşmak için **{n80} bileşen** yeterli."))
    lines.append(I.bullet(f"Yani {len(res['cols'])} değişken, çok az bilgi kaybıyla {n80} boyuta indirgenebilir."))
    # PC1 yükleri
    pc1 = res["loadings"]["PC1"].sort_values(key=abs, ascending=False)
    top = pc1.head(3)
    lines.append(I.bullet("PC1'e en çok katkı veren değişkenler: "
                          + ", ".join(f"{k} ({v:+.2f})" for k, v in top.items())
                          + " → bu bileşen büyük ölçüde bunları temsil ediyor."))
    lines.append(I.bullet("Yüksek |yük| değerleri, o değişkenin ilgili bileşenle güçlü ilişkisini gösterir."))
    return I.joinlines(lines)
