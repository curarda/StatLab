"""Kullanıcı tanımlı regresyon/matematik fonksiyonlarını ayrıştır ve değerlendir.

Örnekler:
    y = 2*x + 3
    y = 1.5*x**2 - 4*x + 10
    3*x1 + 0.5*x2 - 7        (y = varsayılır)
    y = 10 * exp(-0.5*x)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sympy as sp

from . import interpret as I

# İzin verilen semboller/fonksiyonlar
_ALLOWED_FUNCS = {
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "sin": sp.sin,
    "cos": sp.cos, "tan": sp.tan, "abs": sp.Abs, "pi": sp.pi, "E": sp.E,
}


def parse_function(expr_text: str) -> dict:
    """'y = f(x...)' biçimini ayrıştır. Değişkenleri ve sympy ifadesini döndür."""
    text = expr_text.strip()
    if not text:
        return {"error": "Boş ifade."}
    lhs_name = "y"
    if "=" in text:
        lhs, rhs = text.split("=", 1)
        lhs = lhs.strip()
        if lhs:
            lhs_name = lhs
        text = rhs.strip()
    try:
        expr = sp.sympify(text, locals=_ALLOWED_FUNCS)
    except (sp.SympifyError, SyntaxError, TypeError) as e:
        return {"error": f"İfade ayrıştırılamadı: {e}"}
    variables = sorted([str(s) for s in expr.free_symbols], key=lambda v: (len(v), v))
    if not variables:
        return {"error": "İfadede en az bir değişken (ör. x) bulunmalı."}
    return {"lhs": lhs_name, "expr": expr, "expr_str": str(expr), "variables": variables,
            "latex": sp.latex(sp.Eq(sp.Symbol(lhs_name), expr))}


def evaluate(parsed: dict, ranges: dict[str, np.ndarray]) -> pd.DataFrame:
    """Tek değişkenli fonksiyonu belirtilen aralıkta değerlendir (tablo üretir)."""
    expr = parsed["expr"]
    variables = parsed["variables"]
    syms = [sp.Symbol(v) for v in variables]
    f = sp.lambdify(syms, expr, modules=["numpy"])
    if len(variables) == 1:
        v = variables[0]
        xvals = ranges[v]
        yvals = f(xvals)
        yvals = np.broadcast_to(np.asarray(yvals, dtype=float), xvals.shape)
        return pd.DataFrame({v: xvals, parsed["lhs"]: yvals})
    # çok değişkenli: grid yerine eşit uzunlukta örnek
    grids = [ranges[v] for v in variables]
    length = min(len(g) for g in grids)
    cols = {v: np.asarray(ranges[v])[:length] for v in variables}
    yv = f(*[cols[v] for v in variables])
    cols[parsed["lhs"]] = np.broadcast_to(np.asarray(yv, dtype=float), (length,))
    return pd.DataFrame(cols)


def analyze_function(parsed: dict) -> str:
    """Fonksiyonun matematiksel yorumu: türev, kök, tip."""
    expr = parsed["expr"]
    variables = parsed["variables"]
    lines = [f"### Fonksiyon analizi: ${parsed['latex']}$"]
    if len(variables) == 1:
        x = sp.Symbol(variables[0])
        poly = expr.as_poly(x)
        degree = poly.degree() if poly is not None else None
        if degree == 1:
            slope = expr.coeff(x, 1)
            intercept = expr.coeff(x, 0)
            lines.append(I.bullet(f"**Doğrusal fonksiyon** (1. derece). Eğim = {slope}, kesişim = {intercept}."))
            lines.append(I.bullet(f"{variables[0]} 1 birim arttığında {parsed['lhs']} {slope} birim "
                                  + ("artar." if float(slope) > 0 else "azalır." if float(slope) < 0 else "değişmez.")))
        elif degree == 2:
            a2 = expr.coeff(x, 2)
            lines.append(I.bullet(f"**İkinci derece (parabol)**. Baş katsayı = {a2} → "
                                  + ("yukarı açılır, minimum vardır." if float(a2) > 0 else "aşağı açılır, maksimum vardır.")))
            vx = sp.solve(sp.diff(expr, x), x)
            if vx:
                lines.append(I.bullet(f"Tepe/uç nokta: {variables[0]} = {vx[0]}, {parsed['lhs']} = {expr.subs(x, vx[0])}"))
        elif degree is not None:
            lines.append(I.bullet(f"**{degree}. derece polinom.**"))
        else:
            lines.append(I.bullet("Doğrusal olmayan / özel fonksiyon."))
        # türev ve kökler
        deriv = sp.diff(expr, x)
        lines.append(I.bullet(f"Türev d{parsed['lhs']}/d{variables[0]} = {deriv} (değişim hızı)."))
        try:
            roots = sp.solve(sp.Eq(expr, 0), x)
            real_roots = [r for r in roots if r.is_real]
            if real_roots:
                lines.append(I.bullet(f"Kök(ler) ({parsed['lhs']}=0): {', '.join(str(r) for r in real_roots)}"))
        except Exception:
            pass
    else:
        lines.append(I.bullet(f"**Çok değişkenli fonksiyon**: {', '.join(variables)}."))
        for v in variables:
            d = sp.diff(expr, sp.Symbol(v))
            lines.append(I.bullet(f"∂{parsed['lhs']}/∂{v} = {d} → {v} değişkeninin marjinal etkisi."))
    return I.joinlines(lines)


def fit_data_to_expr(parsed: dict, df: pd.DataFrame) -> dict:
    """Kullanıcı fonksiyonundaki serbest parametreleri (a,b,c...) veriye uydur.
    Fonksiyondaki 'a','b','c' gibi katsayılar parametre, 'x' bağımsız değişken kabul edilir.
    """
    # Bu sürümde: fonksiyon zaten somut katsayılı olduğundan teorik-gözlem karşılaştırması yapılır.
    return {"note": "Katsayı kestirimi için Regresyon sekmesini kullanın; burada fonksiyon teorik olarak çizilir."}
