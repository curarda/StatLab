"""Kantitatif finans: getiri/risk, portföy optimizasyonu, CAPM, Monte Carlo, GARCH."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from . import interpret as I

TRADING_DAYS = 252


# ---------------- Veri ----------------

def fetch_prices(symbols: list[str], period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """yfinance ile kapanış fiyatlarını çeker."""
    import yfinance as yf
    data = yf.download(symbols, period=period, interval=interval, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        return pd.DataFrame()
    # 'Close' seviyesini al
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"].copy() if "Close" in data.columns.get_level_values(0) else data.xs("Close", axis=1, level=0)
    else:
        px = data[["Close"]].copy()
        px.columns = symbols[:1]
    px = px.dropna(how="all")
    return px


def to_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    if method == "log":
        r = np.log(prices / prices.shift(1))
    else:
        r = prices.pct_change()
    return r.dropna(how="all")


def _ann_factor(interval_days: int = 1) -> float:
    return TRADING_DAYS / interval_days


# ---------------- Getiri & Risk ----------------

def performance_stats(returns: pd.DataFrame, rf: float = 0.0, periods: int = TRADING_DAYS) -> pd.DataFrame:
    """Her varlık için yıllıklandırılmış getiri/risk metrikleri."""
    rows = []
    rf_p = rf / periods
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 2:
            continue
        ann_ret = r.mean() * periods
        ann_vol = r.std(ddof=1) * np.sqrt(periods)
        sharpe = (r.mean() - rf_p) / r.std(ddof=1) * np.sqrt(periods) if r.std(ddof=1) else np.nan
        downside = r[r < 0].std(ddof=1)
        sortino = (r.mean() - rf_p) / downside * np.sqrt(periods) if downside else np.nan
        cum = (1 + r).cumprod()
        dd = drawdown_series(r)
        mdd = dd.min()
        calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
        var95 = np.percentile(r, 5)
        cvar95 = r[r <= var95].mean()
        rows.append({
            "Varlık": col, "Yıllık Getiri": ann_ret, "Yıllık Volatilite": ann_vol,
            "Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar,
            "Maks. Düşüş": mdd, "VaR %95 (günlük)": var95, "CVaR %95": cvar95,
            "Toplam Getiri": cum.iloc[-1] - 1,
        })
    return pd.DataFrame(rows)


def drawdown_series(r: pd.Series) -> pd.Series:
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    return cum / peak - 1


def value_at_risk(r: pd.Series, level: float = 0.95, method: str = "historical") -> dict:
    r = r.dropna()
    a = 1 - level
    if method == "parametric":
        mu, sd = r.mean(), r.std(ddof=1)
        var = mu + sd * stats.norm.ppf(a)
    else:
        var = np.percentile(r, a * 100)
    cvar = r[r <= var].mean()
    return {"level": level, "method": method, "VaR": float(var), "CVaR": float(cvar)}


def interpret_performance(stats_df: pd.DataFrame, rf: float) -> str:
    if stats_df.empty:
        return "Yeterli veri yok."
    best_sharpe = stats_df.loc[stats_df["Sharpe"].idxmax()]
    worst_dd = stats_df.loc[stats_df["Maks. Düşüş"].idxmin()]
    lines = ["### Getiri & risk değerlendirmesi"]
    lines.append(I.bullet(f"Risksiz oran varsayımı: %{rf*100:.1f} (yıllık)."))
    lines.append(I.bullet(f"**En iyi riske göre getiri:** {best_sharpe['Varlık']} "
                          f"(Sharpe = {best_sharpe['Sharpe']:.2f}, yıllık getiri %{best_sharpe['Yıllık Getiri']*100:.1f})."))
    lines.append(I.bullet(f"**En derin düşüş:** {worst_dd['Varlık']} "
                          f"(maks. düşüş %{worst_dd['Maks. Düşüş']*100:.1f})."))
    lines.append(I.bullet("Sharpe > 1 iyi, > 2 çok iyi kabul edilir. Sortino yalnızca aşağı yönlü riski cezalandırır; "
                          "Calmar getiriyi maksimum düşüşe oranlar."))
    lines.append(I.bullet("VaR %95: normal koşulda günlük kaybın %5 olasılıkla aşabileceği eşik; CVaR o eşik aşıldığında "
                          "beklenen ortalama kayıptır (kuyruk riski)."))
    return I.joinlines(lines)


# ---------------- Portföy optimizasyonu ----------------

def _port_perf(w, mean_r, cov, periods):
    ret = np.dot(w, mean_r) * periods
    vol = np.sqrt(w @ cov @ w) * np.sqrt(periods)
    return ret, vol


def optimize_portfolio(returns: pd.DataFrame, rf: float = 0.0, allow_short: bool = False,
                       periods: int = TRADING_DAYS) -> dict:
    """Minimum-varyans, maksimum-Sharpe portföyleri ve etkin sınır."""
    R = returns.dropna()
    assets = list(R.columns)
    n = len(assets)
    mean_r = R.mean().values
    cov = R.cov().values
    bounds = tuple((-1, 1) if allow_short else (0, 1) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    w0 = np.repeat(1 / n, n)
    rf_p = rf / periods

    def neg_sharpe(w):
        ret, vol = _port_perf(w, mean_r, cov, periods)
        return -(ret - rf) / vol if vol else 1e9

    def port_vol(w):
        return _port_perf(w, mean_r, cov, periods)[1]

    max_sharpe = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
    min_var = minimize(port_vol, w0, method="SLSQP", bounds=bounds, constraints=cons)

    def summarize(res, name):
        w = res.x
        ret, vol = _port_perf(w, mean_r, cov, periods)
        sharpe = (ret - rf) / vol if vol else np.nan
        return {"name": name, "weights": dict(zip(assets, w)), "ret": ret, "vol": vol, "sharpe": sharpe}

    ms = summarize(max_sharpe, "Maksimum Sharpe (teğet)")
    mv = summarize(min_var, "Minimum Varyans")

    # Etkin sınır
    target_rets = np.linspace(mean_r.min() * periods, mean_r.max() * periods, 40)
    frontier = []
    for tr in target_rets:
        c = cons + ({"type": "eq", "fun": lambda w, tr=tr: np.dot(w, mean_r) * periods - tr},)
        res = minimize(port_vol, w0, method="SLSQP", bounds=bounds, constraints=c)
        if res.success:
            frontier.append({"ret": tr, "vol": res.fun})
    return {"assets": assets, "max_sharpe": ms, "min_var": mv,
            "frontier": pd.DataFrame(frontier), "rf": rf,
            "asset_points": [{"Varlık": a, "ret": mean_r[i] * periods,
                              "vol": np.sqrt(cov[i, i]) * np.sqrt(periods)} for i, a in enumerate(assets)]}


RISK_PROFILES = {
    "Muhafazakâr": {"target_vol": 0.08, "desc": "düşük risk, sermaye korumasına öncelik"},
    "Dengeli": {"target_vol": 0.14, "desc": "risk-getiri dengesi"},
    "Büyüme": {"target_vol": 0.20, "desc": "daha yüksek getiri için daha çok risk"},
    "Agresif": {"target_vol": 0.30, "desc": "maksimum büyüme, yüksek oynaklık kabul"},
}


def optimize_for_profile(returns: pd.DataFrame, profile: str = "Dengeli", rf: float = 0.0,
                         allow_short: bool = False, periods: int = TRADING_DAYS) -> dict:
    """Yatırımcı risk profiline göre portföy: hedef oynaklığa göre riskli portföy + risksiz varlık karışımı.

    Teğet (maks-Sharpe) portföy sabittir; profil, risksiz varlık ile teğet portföy arasındaki
    dağılımı (sermaye tahsis doğrusu) belirler. Hedef volatilite düşükse nakit ağırlığı artar.
    """
    base = optimize_portfolio(returns, rf, allow_short, periods)
    tan = base["max_sharpe"]
    target_vol = RISK_PROFILES.get(profile, RISK_PROFILES["Dengeli"])["target_vol"]
    # sermaye tahsisi: w_risky * tan_vol = target_vol  → w_risky = target/tan_vol (0..1, kaldıraçsız)
    w_risky = min(1.0, target_vol / tan["vol"]) if tan["vol"] else 0.0
    w_cash = 1 - w_risky
    port_ret = w_risky * tan["ret"] + w_cash * rf
    port_vol = w_risky * tan["vol"]
    sharpe = (port_ret - rf) / port_vol if port_vol else np.nan
    # her varlığın nihai ağırlığı
    final_w = {a: w_risky * w for a, w in tan["weights"].items()}
    final_w["Nakit / risksiz"] = w_cash
    return {"profile": profile, "target_vol": target_vol, "w_risky": w_risky, "w_cash": w_cash,
            "ret": port_ret, "vol": port_vol, "sharpe": sharpe, "weights": final_w,
            "tangency": tan, "base": base, "rf": rf}


def interpret_profile(res: dict) -> str:
    lines = [f"### {res['profile']} yatırımcı portföyü — yorum katmanı"]
    desc = RISK_PROFILES.get(res["profile"], {}).get("desc", "")
    lines.append(I.bullet(f"Profil hedefi: %{res['target_vol']*100:.0f} yıllık oynaklık ({desc})."))
    lines.append(I.bullet(f"**Dağılım:** %{res['w_risky']*100:.0f} riskli portföy + %{res['w_cash']*100:.0f} nakit/risksiz."))
    lines.append(I.bullet(f"Beklenen: yıllık getiri %{res['ret']*100:.1f}, oynaklık %{res['vol']*100:.1f}, "
                          f"Sharpe {res['sharpe']:.2f}."))
    if res["w_cash"] > 0.4:
        lines.append(I.bullet("Yüksek nakit oranı: düşük risk toleransı sermayenin büyük kısmını risksiz tutuyor "
                              "— getiri sınırlı ama düşüşler yumuşak."))
    elif res["w_cash"] <= 0.01:
        lines.append(I.bullet("Neredeyse tamamı riskli varlıkta: teğet portföyün oynaklığı hedefin altında, "
                              "profil tam kapasite risk alıyor."))
    lines.append(I.bullet("Sermaye tahsis doğrusu (CAL) mantığı: aynı teğet portföy, risksiz varlıkla farklı oranlarda "
                          "karıştırılarak her risk profiline uyarlanır."))
    return I.joinlines(lines)


def risk_parity(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> dict:
    R = returns.dropna()
    cov = R.cov().values
    n = cov.shape[0]

    def obj(w):
        port_var = w @ cov @ w
        mrc = cov @ w
        rc = w * mrc
        target = port_var / n
        return np.sum((rc - target) ** 2)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0.0001, 1) for _ in range(n))
    res = minimize(obj, np.repeat(1 / n, n), method="SLSQP", bounds=bounds, constraints=cons)
    w = res.x
    return {"weights": dict(zip(R.columns, w)),
            "vol": float(np.sqrt(w @ cov @ w) * np.sqrt(periods))}


def interpret_optimization(res: dict) -> str:
    ms, mv = res["max_sharpe"], res["min_var"]
    lines = ["### Portföy optimizasyonu (Markowitz)"]
    lines.append(I.bullet(f"**Maksimum Sharpe (teğet) portföy** — yıllık getiri %{ms['ret']*100:.1f}, "
                          f"volatilite %{ms['vol']*100:.1f}, Sharpe {ms['sharpe']:.2f}."))
    top = sorted(ms["weights"].items(), key=lambda kv: -kv[1])
    lines.append(I.bullet("Ağırlıklar: " + ", ".join(f"{a} %{w*100:.1f}" for a, w in top if abs(w) > 0.005)))
    lines.append(I.bullet(f"**Minimum varyans portföy** — volatilite %{mv['vol']*100:.1f} (en düşük risk), "
                          f"yıllık getiri %{mv['ret']*100:.1f}."))
    lines.append(I.bullet("Etkin sınır: her risk düzeyi için ulaşılabilir en yüksek getiriyi veren portföyler. "
                          "Teğet portföy, risksiz varlıkla birleştirildiğinde en yüksek Sharpe'ı sunar."))
    lines.append(I.bullet("Not: Geçmiş getiri/kovaryansa dayanır; gelecekte aynen sürmeyebilir. Aşırı yoğun ağırlıklara dikkat."))
    return I.joinlines(lines)


# ---------------- CAPM / faktör ----------------

def capm(asset_ret: pd.Series, market_ret: pd.Series, rf: float = 0.0,
         periods: int = TRADING_DAYS, alpha_level: float = 0.05) -> dict:
    import statsmodels.api as sm
    rf_p = rf / periods
    pair = pd.concat([asset_ret, market_ret], axis=1).dropna()
    y = pair.iloc[:, 0] - rf_p
    x = pair.iloc[:, 1] - rf_p
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]
    return {"alpha_daily": float(alpha), "alpha_annual": float(alpha * periods),
            "beta": float(beta), "alpha_p": float(model.pvalues.iloc[0]),
            "beta_p": float(model.pvalues.iloc[1]), "r2": float(model.rsquared),
            "n": int(model.nobs), "alpha_level": alpha_level}


def rolling_beta(asset_ret: pd.Series, market_ret: pd.Series, window: int = 60) -> pd.Series:
    pair = pd.concat([asset_ret, market_ret], axis=1).dropna()
    cov = pair.iloc[:, 0].rolling(window).cov(pair.iloc[:, 1])
    var = pair.iloc[:, 1].rolling(window).var()
    return (cov / var).dropna()


def factor_regression(asset_ret: pd.Series, factors: pd.DataFrame, rf: float = 0.0,
                      periods: int = TRADING_DAYS) -> dict:
    import statsmodels.api as sm
    pair = pd.concat([asset_ret, factors], axis=1).dropna()
    y = pair.iloc[:, 0] - rf / periods
    X = sm.add_constant(pair[factors.columns])
    model = sm.OLS(y, X).fit()
    rows = []
    for name in model.params.index:
        rows.append({"Faktör": "alpha" if name == "const" else name,
                     "Katsayı": float(model.params[name]), "t": float(model.tvalues[name]),
                     "p": float(model.pvalues[name]), "Anlam": I.stars(float(model.pvalues[name])) or "—"})
    return {"table": pd.DataFrame(rows), "r2": float(model.rsquared),
            "alpha_annual": float(model.params.get("const", 0) * periods), "n": int(model.nobs)}


def interpret_capm(res: dict) -> str:
    a = res["alpha_level"]
    lines = ["### CAPM regresyonu"]
    lines.append(I.bullet(f"**Beta = {res['beta']:.3f}** (p={res['beta_p']:.4g}) → "
                          + ("piyasadan daha oynak (agresif)." if res['beta'] > 1
                             else "piyasadan daha az oynak (defansif)." if res['beta'] < 1 else "piyasayla aynı.")))
    lines.append(I.bullet(f"Piyasa 1 birim yükseldiğinde varlık ~{res['beta']:.2f} birim hareket eder."))
    sig = "**anlamlı pozitif (piyasa üstü getiri)**" if (res['alpha_annual'] > 0 and res['alpha_p'] < a) else \
          "**anlamlı negatif**" if (res['alpha_annual'] < 0 and res['alpha_p'] < a) else "sıfırdan farksız (anlamlı değil)"
    lines.append(I.bullet(f"**Alpha (yıllık) = %{res['alpha_annual']*100:.2f}** (p={res['alpha_p']:.4g}) → {sig}. "
                          "Pozitif ve anlamlı alpha, riske göre düzeltilmiş fazladan getiri demektir."))
    lines.append(I.bullet(f"R² = {res['r2']:.3f} → getirinin %{res['r2']*100:.1f}'i piyasa hareketiyle açıklanıyor "
                          "(kalanı varlığa özgü/sistematik olmayan risk)."))
    return I.joinlines(lines)


# ---------------- Monte Carlo ----------------

def monte_carlo(returns: pd.Series, init: float = 10000, horizon: int = 252,
                n_sims: int = 1000, periods: int = TRADING_DAYS) -> dict:
    r = returns.dropna()
    mu, sd = r.mean(), r.std(ddof=1)
    sims = np.zeros((n_sims, horizon + 1))
    sims[:, 0] = init
    rnd = np.random.default_rng(42).normal(mu, sd, (n_sims, horizon))
    for t in range(1, horizon + 1):
        sims[:, t] = sims[:, t - 1] * (1 + rnd[:, t - 1])
    final = sims[:, -1]
    return {"paths": sims, "final": final, "init": init, "horizon": horizon,
            "p5": float(np.percentile(final, 5)), "p50": float(np.percentile(final, 50)),
            "p95": float(np.percentile(final, 95)), "mean": float(final.mean()),
            "prob_loss": float(np.mean(final < init)), "n_sims": n_sims}


def interpret_montecarlo(res: dict, goal: float | None = None) -> str:
    lines = [f"### Monte Carlo simülasyonu ({res['n_sims']} senaryo, {res['horizon']} gün)"]
    lines.append(I.bullet(f"Başlangıç {res['init']:,.0f} → medyan sonuç **{res['p50']:,.0f}** "
                          f"(%5–%95 aralığı: {res['p5']:,.0f} – {res['p95']:,.0f})."))
    lines.append(I.bullet(f"Zarar (başlangıcın altına düşme) olasılığı: **%{res['prob_loss']*100:.1f}**."))
    if goal:
        prob = float(np.mean(res["final"] >= goal))
        lines.append(I.bullet(f"**{goal:,.0f} hedefine ulaşma olasılığı: %{prob*100:.1f}.**"))
    lines.append(I.bullet("Geçmiş ortalama/volatiliteye dayalı rastgele yollar üretir; getiriler normal ve sabit "
                          "varsayılır (gerçekte kuyruk riski daha yüksek olabilir)."))
    return I.joinlines(lines)


# ---------------- Backtesting ----------------

def backtest(returns: pd.DataFrame, weights: dict, rebalance: str = "M",
             benchmark: str | None = None, init: float = 10000, periods: int = TRADING_DAYS) -> dict:
    """Verilen ağırlıklarla portföyü geçmişte simüle et (periyodik yeniden dengeleme)."""
    R = returns.dropna().copy()
    assets = [a for a in weights if a in R.columns and a != benchmark]
    w = np.array([weights[a] for a in assets], dtype=float)
    w = w / w.sum() if w.sum() else w
    Ra = R[assets]
    # yeniden dengeleme dönemleri
    if rebalance == "Buy&Hold":
        reb_idx = set()
    else:
        marks = R.resample(rebalance).last().index
        reb_idx = set(R.index.searchsorted(marks))
    port_ret = []
    cur_w = w.copy()
    for i, (_, row) in enumerate(Ra.iterrows()):
        if i in reb_idx:
            cur_w = w.copy()
        r = float(np.dot(cur_w, row.values))
        port_ret.append(r)
        # ağırlıkları getiriyle güncelle (drift)
        cur_w = cur_w * (1 + row.values)
        cur_w = cur_w / cur_w.sum() if cur_w.sum() else cur_w
    port_ret = pd.Series(port_ret, index=Ra.index)
    equity = init * (1 + port_ret).cumprod()
    metrics = _perf_metrics(port_ret, periods)
    out = {"equity": equity, "returns": port_ret, "metrics": metrics, "init": init,
           "assets": assets, "rebalance": rebalance}
    if benchmark and benchmark in R.columns:
        bench_ret = R[benchmark]
        bench_eq = init * (1 + bench_ret).cumprod()
        active = port_ret - bench_ret
        te = active.std(ddof=1) * np.sqrt(periods)
        ir = active.mean() / active.std(ddof=1) * np.sqrt(periods) if active.std(ddof=1) else np.nan
        out["benchmark"] = benchmark
        out["bench_equity"] = bench_eq
        out["bench_metrics"] = _perf_metrics(bench_ret, periods)
        out["tracking_error"] = float(te)
        out["info_ratio"] = float(ir)
    return out


def _perf_metrics(r: pd.Series, periods: int) -> dict:
    ann_ret = (1 + r).prod() ** (periods / len(r)) - 1 if len(r) else np.nan
    vol = r.std(ddof=1) * np.sqrt(periods)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(periods) if r.std(ddof=1) else np.nan
    mdd = drawdown_series(r).min()
    return {"CAGR": float(ann_ret), "vol": float(vol), "sharpe": float(sharpe),
            "max_dd": float(mdd), "total": float((1 + r).prod() - 1)}


def interpret_backtest(res: dict) -> str:
    m = res["metrics"]
    lines = [f"### Backtest sonucu ({res['rebalance']} yeniden dengeleme)"]
    lines.append(I.bullet(f"Toplam getiri %{m['total']*100:.1f} · Yıllık (CAGR) %{m['CAGR']*100:.1f} · "
                          f"volatilite %{m['vol']*100:.1f} · Sharpe {m['sharpe']:.2f} · maks. düşüş %{m['max_dd']*100:.1f}."))
    if "benchmark" in res:
        bm = res["bench_metrics"]
        diff = m["CAGR"] - bm["CAGR"]
        lines.append(I.bullet(f"Benchmark ({res['benchmark']}) CAGR %{bm['CAGR']*100:.1f} → portföy "
                              f"**{diff*100:+.1f} puan** {'üstün' if diff>0 else 'geride'}."))
        lines.append(I.bullet(f"İzleme hatası %{res['tracking_error']*100:.1f}, **bilgi oranı {res['info_ratio']:.2f}** "
                              "(aktif getirinin riske göre kalitesi; >0.5 iyi, >1 çok iyi)."))
    lines.append(I.bullet("Backtest geçmiş performanstır; geleceği garanti etmez. İşlem maliyeti/kayma dahil değildir "
                          "(gerçek getiri biraz daha düşük olur)."))
    return I.joinlines(lines)


# ---------------- Tahvil matematiği ----------------

def bond_analytics(face: float, coupon_rate: float, years: float, ytm: float, freq: int = 2) -> dict:
    """Tahvil fiyatı, Macaulay & değiştirilmiş süre (duration), konveksite."""
    n = int(round(years * freq))
    c = face * coupon_rate / freq         # dönemsel kupon
    y = ytm / freq                        # dönemsel getiri
    times = np.arange(1, n + 1)
    cfs = np.full(n, c, dtype=float)
    cfs[-1] += face                       # anapara son dönemde
    disc = (1 + y) ** times
    pv = cfs / disc
    price = pv.sum()
    mac_dur = (times * pv).sum() / price / freq          # yıl
    mod_dur = mac_dur / (1 + y)
    convexity = (pv * times * (times + 1)).sum() / (price * (1 + y) ** 2) / (freq ** 2)
    current_yield = (face * coupon_rate) / price
    return {"price": float(price), "face": face, "coupon_rate": coupon_rate, "years": years,
            "ytm": ytm, "freq": freq, "macaulay": float(mac_dur), "modified": float(mod_dur),
            "convexity": float(convexity), "current_yield": float(current_yield)}


def bond_price_change(res: dict, dy: float) -> float:
    """Getiri dy kadar değişince fiyatta yaklaşık % değişim (süre + konveksite)."""
    return -res["modified"] * dy + 0.5 * res["convexity"] * dy ** 2


def interpret_bond(res: dict) -> str:
    lines = ["### Tahvil analizi"]
    lines.append(I.bullet(f"Fiyat = {res['price']:.2f} (nominal {res['face']}). "
                          + ("Primli (nominalin üstünde) — kupon > YTM." if res['price'] > res['face']
                             else "İskontolu (nominalin altında) — kupon < YTM." if res['price'] < res['face']
                             else "Başabaş.")))
    lines.append(I.bullet(f"Cari getiri %{res['current_yield']*100:.2f}, YTM %{res['ytm']*100:.2f}."))
    lines.append(I.bullet(f"**Değiştirilmiş süre (duration) = {res['modified']:.2f}** → faiz %1 (100 bp) artarsa "
                          f"fiyat yaklaşık **%{res['modified']:.2f} düşer** (konveksite bunu hafif yumuşatır)."))
    lines.append(I.bullet(f"Macaulay süresi {res['macaulay']:.2f} yıl, konveksite {res['convexity']:.2f}."))
    dy = 0.01
    ch = bond_price_change(res, dy) * 100
    lines.append(I.bullet(f"Örnek: getiri +100 bp → fiyat ≈ %{ch:.2f} değişir."))
    lines.append(I.bullet("Uzun süreli tahviller faize daha duyarlıdır (daha çok risk ve potansiyel getiri)."))
    return I.joinlines(lines)


# ---------------- Makro & faiz ----------------

# ABD Hazine getiri endeksleri (yfinance): süre → sembol
RATE_TICKERS = {"3 Ay": "^IRX", "5 Yıl": "^FVX", "10 Yıl": "^TNX", "30 Yıl": "^TYX"}


def fetch_macro(period: str = "2y") -> dict:
    """Getiri eğrisi noktaları (^IRX/^FVX/^TNX/^TYX) ve VIX'i çeker."""
    import yfinance as yf
    tickers = list(RATE_TICKERS.values()) + ["^VIX"]
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"]
    else:
        px = data.to_frame()
    px = px.dropna(how="all")
    # ^IRX/^TNX vb. yüzde cinsindendir (ör. 4.35 = %4.35)
    yields = {label: float(px[sym].dropna().iloc[-1]) for label, sym in RATE_TICKERS.items() if sym in px}
    vix = float(px["^VIX"].dropna().iloc[-1]) if "^VIX" in px and px["^VIX"].notna().any() else np.nan
    # eğim: 10Y - 3M
    slope = yields.get("10 Yıl", np.nan) - yields.get("3 Ay", np.nan)
    hist = px.rename(columns={v: k for k, v in {**RATE_TICKERS, "^VIX": "VIX"}.items()})
    return {"yields": yields, "vix": vix, "slope": float(slope), "history": hist,
            "asof": str(px.index[-1].date()) if len(px) else "—"}


def interpret_macro(res: dict) -> str:
    y = res["yields"]
    lines = [f"### Makro & faiz görünümü ({res['asof']}) — yorum katmanı"]
    if y:
        lines.append(I.bullet("Getiri eğrisi: " + ", ".join(f"{k} %{v:.2f}" for k, v in y.items())))
    s = res["slope"]
    if s == s:
        if s < 0:
            lines.append(I.bullet(f"**Getiri eğrisi ters (10Y−3M = {s:+.2f} puan)** → tarihsel olarak güçlü bir "
                                  "resesyon sinyali; kısa vade uzun vadeden yüksek getiriyor."))
        elif s < 0.5:
            lines.append(I.bullet(f"Eğri düz (10Y−3M = {s:+.2f}) → büyümede belirsizlik/yavaşlama beklentisi."))
        else:
            lines.append(I.bullet(f"Eğri normal/dik (10Y−3M = {s:+.2f}) → sağlıklı büyüme beklentisi."))
    v = res["vix"]
    if v == v:
        vq = ("düşük — piyasa sakin/rehavet" if v < 15 else "normal" if v < 20
              else "yüksek — tedirginlik" if v < 30 else "çok yüksek — panik/kriz")
        lines.append(I.bullet(f"VIX (korku endeksi) = {v:.1f} → {vq}."))
    lines.append(I.bullet("Faizler tüm varlık fiyatlamasının çıpasıdır: yüksek faiz tahvil getirisini artırır, "
                          "hisse değerlemesini baskılar. Eğri eğimi ve VIX, piyasanın **ileriye dönük beklentisini** yansıtır."))
    lines.append(I.bullet("Not: Bunlar kantitatif piyasa göstergeleridir; siyasi/haber yorumu için bağlamı Claude katmanına ekleyin."))
    return I.joinlines(lines)


# ---------------- GARCH ----------------

def garch_volatility(returns: pd.Series, p: int = 1, q: int = 1, horizon: int = 10) -> dict:
    from arch import arch_model
    r = returns.dropna() * 100  # yüzde ölçek
    am = arch_model(r, vol="GARCH", p=p, q=q, dist="normal")
    fit = am.fit(disp="off")
    cond_vol = fit.conditional_volatility / 100
    fc = fit.forecast(horizon=horizon, reindex=False)
    fvar = np.sqrt(fc.variance.values[-1]) / 100
    return {"cond_vol": cond_vol, "forecast_vol": fvar, "params": fit.params.to_dict(),
            "aic": float(fit.aic), "horizon": horizon, "p": p, "q": q}


def interpret_garch(res: dict) -> str:
    lines = [f"### GARCH({res['p']},{res['q']}) oynaklık modeli"]
    a1 = res["params"].get("alpha[1]", np.nan)
    b1 = res["params"].get("beta[1]", np.nan)
    persist = (a1 + b1) if (a1 == a1 and b1 == b1) else np.nan
    lines.append(I.bullet(f"AIC = {res['aic']:.1f}. Kalıcılık (α+β) = {persist:.3f} "
                          + ("→ **yüksek**: şoklar uzun süre etkili, oynaklık kümelenmesi belirgin." if persist > 0.9
                             else "→ oynaklık şokları görece hızlı sönümleniyor.")))
    lines.append(I.bullet(f"Sonraki {res['horizon']} gün için öngörülen günlük oynaklık "
                          f"%{res['forecast_vol'][0]*100:.2f} → %{res['forecast_vol'][-1]*100:.2f} bandında."))
    lines.append(I.bullet("GARCH, sabit varyans varsayımını gevşetir; risk yönetimi ve VaR için gerçekçi, "
                          "zamanla değişen oynaklık verir."))
    return I.joinlines(lines)
