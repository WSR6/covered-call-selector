# covered_call_app.py
import streamlit as st
import yfinance as yf
from datetime import datetime
from scipy.stats import norm
import math
import pandas as pd
import time
import logging

# ------------------------------------------------------------------
# Silence yfinance noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ------------------------------------------------------------------
# Core helpers (same as your desktop app)
# ------------------------------------------------------------------
def days_to_expiry(expiry: str) -> int:
    today = datetime.now().date()
    exp = datetime.strptime(expiry, "%Y-%m-%d").date()
    return (exp - today).days


def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1)


def price(symbol: str) -> float:
    t = yf.Ticker(symbol)
    p = t.info.get("regularMarketPrice") or t.info.get("currentPrice")
    if not p:
        raise ValueError("Price fetch failed.")
    return float(p)


def vix_value() -> float:
    v = yf.Ticker("^VIX")
    val = v.info.get("regularMarketPrice") or v.info.get("currentPrice")
    if not val:
        raise ValueError("VIX fetch failed.")
    return float(val)


def risk_free_rate() -> float:
    irx = yf.Ticker("^IRX")
    rate = irx.info.get("regularMarketPrice") or irx.info.get("currentPrice")
    return (rate / 100) if rate else 0.04


def rsi_14(symbol: str) -> float | None:
    try:
        hist = yf.download(
            symbol,
            period="1mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if len(hist) < 15:
            return None
        delta = hist["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        val = float(rsi.iloc[-1]) if not math.isnan(rsi.iloc[-1]) else None
        return val
    except Exception:
        return None


def score(opt: dict) -> float:
    ideal_otm, ideal_weeks = 7.5, 4.5
    ideal_delta, ideal_ret = 0.25, 6.5
    otm = abs(opt["otm_pct"] - ideal_otm) / 5.0
    wks = abs(opt["days_to_exp"] / 7 - ideal_weeks) / 3.0
    dlt = abs(opt["delta"] - ideal_delta) / 0.1
    ret = abs(opt["premium_pct"] - ideal_ret) / 7.0
    return (otm + wks + dlt + ret) / 4.0


def find_options(
    symbol: str = "TSLA",
    otm_min: float = 5,
    otm_max: float = 10,
    weeks_min: int = 3,
    weeks_max: int = 6,
    delta_min: float = 0.2,
    delta_max: float = 0.3,
    premium_min: float = 2,
    return_min: float = 3,
    return_max: float = 10,
    vix_thr: float = 20,
    prem_type: str = "bid",
):
    # ---- retry on temporary Yahoo JSON glitches ----
    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            S = price(symbol)
            V = vix_value()
            r = risk_free_rate()
            break
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(5)  # wait 5 s before retry
    else:
        raise RuntimeError("Failed after retries")

    dmin, dmax = weeks_min * 7, weeks_max * 7
    matches = []

    for exp in ticker.options:
        days = days_to_expiry(exp)
        if not (dmin <= days <= dmax):
            continue
        chain = ticker.option_chain(exp)
        for _, row in chain.calls.iterrows():
            K = row["strike"]
            otm_pct = (K - S) / S * 100
            if not (otm_min <= otm_pct <= otm_max):
                continue
            sigma = row["impliedVolatility"]
            T = days / 365.0
            delta = call_delta(S, K, T, r, sigma)
            if not (delta_min <= delta <= delta_max):
                continue
            premium = (
                row["bid"]
                if prem_type == "bid"
                else row["ask"]
                if prem_type == "ask"
                else (row["bid"] + row["ask"]) / 2
            )
            prem_pct = premium / S * 100
            if prem_pct < premium_min or not (return_min <= prem_pct <= return_max):
                continue
            opt = {
                "expiry": exp,
                "strike": K,
                "delta": delta,
                "premium": premium,
                "premium_pct": prem_pct,
                "otm_pct": otm_pct,
                "days_to_exp": days,
                "breakeven": S + premium,
            }
            opt["score"] = score(opt)
            matches.append(opt)

    matches.sort(key=lambda x: x["score"])
    return S, V, matches, vix_thr


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Covered Call Selector", layout="wide")
st.title("Covered Call Selector – Mobile Ready")

col1, col2 = st.columns(2)
with col1:
    symbol = st.text_input("Symbol", value="TSLA").strip().upper()
    otm_min = st.number_input("OTM % Min", value=5.0, step=0.1)
    otm_max = st.number_input("OTM % Max", value=10.0, step=0.1)
    weeks_min = st.number_input("Weeks Min", value=3, step=1)
    weeks_max = st.number_input("Weeks Max", value=6, step=1)
with col2:
    delta_min = st.number_input("Delta Min", value=0.2, step=0.01)
    delta_max = st.number_input("Delta Max", value=0.3, step=0.01)
    premium_min = st.number_input("Premium % Min", value=2.0, step=0.1)
    return_min = st.number_input("Return % Min", value=3.0, step=0.1)
    return_max = st.number_input("Return % Max", value=10.0, step=0.1)
    vix_thr = st.number_input("VIX Threshold", value=20.0, step=0.5)
    prem_type = st.selectbox("Premium Type", ["bid", "ask", "mid"])

if st.button("Refresh", type="primary"):
    with st.spinner("Fetching data…"):
        try:
            S, V, matches, _ = find_options(
                symbol, otm_min, otm_max, weeks_min, weeks_max,
                delta_min, delta_max, premium_min, return_min, return_max,
                vix_thr, prem_type
            )
            rsi_val = rsi_14(symbol)

            # ---- results ----
            st.success(f"**{symbol}** price: **${S:,.2f}** | VIX: **{V:.2f}**")
            if rsi_val is not None:
                rsi_txt = f"RSI (14): **{rsi_val:.1f}**"
                if rsi_val > 70:
                    rsi_txt += " 🟢 Overbought – good for calls"
                elif rsi_val < 30:
                    rsi_txt += " 🔴 Oversold – avoid"
                else:
                    rsi_txt += " ⚪ Neutral"
                st.markdown(rsi_txt)

            if V <= vix_thr:
                st.warning(f"VIX {V:.1f} ≤ {vix_thr} → rule says **wait**. Options shown anyway.")

            if matches:
                for o in matches:
                    tag = "🟢" if o["score"] < 0.3 else "🟡" if o["score"] < 0.6 else "🔴"
                    with st.expander(f"{tag} {o['expiry']} ({o['days_to_exp']} d) – Score {o['score']:.2f}"):
                        st.write(f"**Strike** ${o['strike']:,.2f} ({o['otm_pct']:.1f}% OTM)")
                        st.write(f"**Δ** {o['delta']:.2f} | **Prem** ${o['premium']:.2f} ({o['premium_pct']:.2f}%)")
                        st.write(f"**Break-even** ${o['breakeven']:,.2f}")
            else:
                st.info("No options match the criteria.")

            st.caption("Re-invest ~50 % of premium, keep the rest for taxes.")

            # ---- CSV export ----
            if matches:
                df = pd.DataFrame(matches)
                csv = df.to_csv(index=False).encode()
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"{symbol}_calls.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error: {e}")

st.sidebar.markdown("### Settings")
st.sidebar.caption("All inputs are saved in your browser session. Refresh to re-run.")