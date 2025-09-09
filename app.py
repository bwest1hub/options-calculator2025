import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

st.title("Call Option Calculator")

# --- Input parameters ---
st.header("Option Parameters")
S = st.number_input("Current Stock Price (S)", value=100.0, min_value=0.01, format="%f")
K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, format="%f")
premium = st.number_input("Option Premium (call price)", value=5.0, min_value=0.0, format="%f")
sigma = st.number_input("Implied Volatility (σ, in %)", value=20.0, min_value=0.0, format="%f")
r = st.number_input("Risk-free Interest Rate (r, in %)", value=1.0, format="%f")
t_days = st.number_input("Days to Expiration", value=30, min_value=1, format="%d")

# Convert inputs
T = t_days / 365.0          # time to expiration in years
vol = sigma / 100.0         # volatility in decimal
rate = r / 100.0            # risk-free rate in decimal

# --- Black-Scholes Calculations ---
if S > 0 and K > 0 and T > 0 and vol > 0:
    d1 = (np.log(S / K) + (rate + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    call_price_bs = S * norm.cdf(d1) - K * np.exp(-rate * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    theta = -(S * norm.pdf(d1) * vol) / (2 * np.sqrt(T)) - rate * K * np.exp(-rate * T) * norm.cdf(d2)
    theta = theta / 365.0         # per day
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0
    rho = K * T * np.exp(-rate * T) * norm.cdf(d2) / 100.0
else:
    call_price_bs = max(S - K, 0)
    delta = 1.0 if S > K else 0.0
    gamma = 0.0
    theta = 0.0
    vega = 0.0
    rho = 0.0

break_even = K + premium

# --- Display Results ---
st.subheader("Option Metrics (Black-Scholes)")
st.write(f"Break-even Price: {break_even:.2f}")
st.write("Max Profit: Unlimited")
st.write(f"Max Loss: {premium:.2f}")
st.write(f"Theoretical Call Price (BS): {call_price_bs:.2f}")
st.write(f"Delta (Δ): {delta:.4f}")
st.write(f"Gamma (Γ): {gamma:.4f}")
st.write(f"Theta (Θ): {theta:.4f} per day")
st.write(f"Vega: {vega:.4f} per 1% vol")
st.write(f"Rho: {rho:.4f} per 1% rate")

# --- Scenario Analysis ---
st.header("Scenario Analysis")
moves = []
for i in range(4):
    move = st.number_input(f"Stock Move Scenario {i+1} (%)", value=0.0, format="%f", key=f"move{i}")
    moves.append(move)

scenario_prices = [(1 + m/100.0) * S for m in moves]
intrinsic_values = [max(price - K, 0) for price in scenario_prices]
profit_losses = [intrinsic - premium for intrinsic in intrinsic_values]

scenario_df = pd.DataFrame({
    "Stock Move (%)": moves,
    "Price at Expiry": scenario_prices,
    "Option Payoff": intrinsic_values,
    "Profit / Loss": profit_losses
}).round(2)

st.subheader("Scenario P/L Table")
st.write(scenario_df)

# --- Payoff Chart ---
st.header("Payoff Chart at Expiration")
min_price = 0.0
max_price = max(S * 2, K * 2, break_even * 1.5)
prices = np.linspace(min_price, max_price, 500)
payoff = np.maximum(prices - K, 0)
profit = payoff - premium

fig, ax = plt.subplots()
ax.plot(prices, payoff, label="Option Payoff", color='tab:blue')
ax.plot(prices, profit, label="Profit / Loss", color='tab:orange')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(K, color='gray', linestyle='--', label="Strike Price")
ax.set_xlabel("Stock Price at Expiration")
ax.set_ylabel("Value")
ax.set_title("Call Option Payoff vs Profit at Expiration")
ax.legend()
st.pyplot(fig)
plt.close(fig)
