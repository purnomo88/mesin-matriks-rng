import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from itertools import product
from scipy.stats import chisquare

st.set_page_config(page_title="4D Probabilistic Strength Analyzer", layout="wide")

# GANTI LINK DI BAWAH DENGAN LINK CSV GOOGLE SHEETS ANDA
GOOGLE_SHEETS_CSV_URL = "https://docs.google.com/spreadsheets/d/1prsu_8P8rxoKluOdbozwPrCdtJmGd9kBoqzfDnqVlVU/export?format=csv&gid=0"

st.title("Multi-Signal 4D Probability Engine")
st.warning(
    "Statistical Disclaimer: This system combines Global Frequency, Recency-Weighting, "
    "and Markov Transition (H to H+1) into a single composite score. This is NOT a guaranteed "
    "prediction. If your data passes the chi-square test (distribution close to uniform), all "
    "scores below are mathematically equivalent to random guessing - useful only as a historical "
    "ranking, not an absolute forecast."
)


@st.cache_data(ttl=600, show_spinner=True)
def load_data(source):
    try:
        df = pd.read_csv(source, header=None)
    except Exception as e:
        raise ValueError(f"Failed to read data source: {e}")

    if df.empty:
        raise ValueError("The data source is empty.")

    df = df.iloc[1:]

    cells = []
    for row in df.itertuples(index=False):
        for val in row:
            if pd.notna(val):
                cells.append(str(val))

    flat_text = " ".join(cells)
    tokens = re.findall(r"\d{4}", flat_text)

    if len(tokens) < 10:
        digits = re.findall(r"\d", flat_text)
        usable = len(digits) - (len(digits) % 4)
        tokens = ["".join(digits[i:i + 4]) for i in range(0, usable, 4)]

    if len(tokens) < 10:
        raise ValueError("Not enough valid 4-digit entries found (minimum 10 required).")

    return np.array(tokens, dtype=str)


def chi_square_sanity_check(history: np.ndarray):
    n = len(history)
    results = []
    for pos in range(4):
        digits = [int(s[pos]) for s in history]
        obs = np.bincount(digits, minlength=10)
        exp = np.full(10, n / 10)
        chi2, p = chisquare(obs, exp)
        results.append({
            "Position": f"Digit {pos + 1}",
            "Chi2": round(chi2, 2),
            "p-value": round(p, 4),
            "Status": "Random (Normal)" if p > 0.05 else "Significant Bias Detected"
        })
    return pd.DataFrame(results)


def build_pairs(history: np.ndarray):
    return list(zip(history[:-1], history[1:]))


def per_position_scores(history: np.ndarray, pairs, baseline: str, recency_halflife: int = 50):
    n = len(history)
    scores = []

    for pos in range(4):
        digit_series = np.array([int(s[pos]) for s in history])

        global_freq = np.bincount(digit_series, minlength=10) / n

        weights = np.exp(-np.arange(n)[::-1] / recency_halflife)
        weighted_freq = np.zeros(10)
        for d in range(10):
            weighted_freq[d] = weights[digit_series == d].sum()
        weighted_freq = weighted_freq / weighted_freq.sum()

        markov_counter = Counter()
        for today, tomorrow in pairs:
            if today[pos] == baseline[pos]:
                markov_counter[int(tomorrow[pos])] += 1

        total_markov = sum(markov_counter.values())
        markov_freq = np.zeros(10)
        if total_markov > 0:
            for d, c in markov_counter.items():
                markov_freq[d] = c / total_markov

        composite = (0.3 * global_freq) + (0.3 * weighted_freq) + (0.4 * markov_freq)

        df_pos = pd.DataFrame({
            "Digit": range(10),
            "Global_Freq": global_freq.round(4),
            "Recency_Freq": weighted_freq.round(4),
            "Markov_Freq": markov_freq.round(4),
            "Composite_Score": composite.round(4)
        }).sort_values("Composite_Score", ascending=False).reset_index(drop=True)

        scores.append(df_pos)

    return scores


def full_2d_markov(pairs, target_2d, top_n=5):
    counter = Counter()
    for today, tomorrow in pairs:
        if today[2:] == target_2d:
            counter[tomorrow[2:]] += 1
    return counter.most_common(top_n)


def generate_strong_numbers(scores, top_k=(2, 2, 3, 3)):
    top_per_pos = []
    for i, df_pos in enumerate(scores):
        top_per_pos.append(list(df_pos.head(top_k[i])["Digit"].astype(str)))

    combos = []
    for d1, d2, d3, d4 in product(*top_per_pos):
        s1 = scores[0].set_index("Digit").loc[int(d1), "Composite_Score"]
        s2 = scores[1].set_index("Digit").loc[int(d2), "Composite_Score"]
        s3 = scores[2].set_index("Digit").loc[int(d3), "Composite_Score"]
        s4 = scores[3].set_index("Digit").loc[int(d4), "Composite_Score"]

        combos.append({
            "4D": f"{d1}{d2}{d3}{d4}",
            "Back_2D": f"{d3}{d4}",
            "Total_Score": round(s1 + s2 + s3 + s4, 4)
        })

    combos = pd.DataFrame(combos).sort_values("Total_Score", ascending=False).reset_index(drop=True)
    return combos


def main():
    st.sidebar.header("Settings")
    recency_halflife = st.sidebar.slider(
        "Recency Half-life (smaller = more focus on recent data)", 10, 200, 50
    )

    if not GOOGLE_SHEETS_CSV_URL.strip() or GOOGLE_SHEETS_CSV_URL == "PASTE_LINK_CSV_GOOGLE_SHEETS_DI_SINI":
        st.error("Please fill GOOGLE_SHEETS_CSV_URL in app.py first.")
        return

    try:
        history = load_data(GOOGLE_SHEETS_CSV_URL.strip())
    except ValueError as ve:
        st.error(f"Data Error: {ve}")
        return
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        return

    st.success(f"{len(history)} historical 4D entries loaded successfully.")
    st.caption("Data source: Google Sheets CSV URL")

    with st.expander("Sanity Check: Chi-Square Test (Is This Data Truly Random?)"):
        st.dataframe(chi_square_sanity_check(history), use_container_width=True)
        st.caption(
            "If p-value > 0.05 for all positions, the data is consistent with a pure RNG - "
            "meaning the scores below are historical-descriptive only, not absolute predictions."
        )

    pairs = build_pairs(history)

    st.divider()
    st.subheader("Enter Yesterday's Number (Baseline H)")
    baseline = st.text_input("Enter yesterday's 4-digit result (e.g. 3506):", max_chars=4)

    calc = st.button("Calculate Today's Strong Numbers (H+1)", type="primary")

    if not calc:
        return

    baseline = baseline.strip()
    if not baseline.isdigit() or len(baseline) != 4:
        st.error("Baseline must be exactly 4 numeric digits (e.g. 3506).")
        return

    try:
        scores = per_position_scores(history, pairs, baseline, recency_halflife)
    except Exception as e:
        st.error(f"Calculation Error: {e}")
        return

    st.divider()
    st.header("Composite Score per Position")
    labels = ["Digit 1 (Thousands)", "Digit 2 (Hundreds)", "Digit 3 (Tens)", "Digit 4 (Units)"]
    cols = st.columns(4)

    for i, (label, df_pos) in enumerate(zip(labels, scores)):
        with cols[i]:
            st.markdown(f"**{label}**")
            st.dataframe(df_pos.head(5), use_container_width=True, hide_index=True)

    st.divider()
    st.header("Priority: Back 2D (Tens + Units)")
    target_2d = baseline[2:]
    full_2d = full_2d_markov(pairs, target_2d, top_n=5)

    if full_2d:
        st.success(f"Historical transition matches found for 2D baseline '{target_2d}':")
        st.table(pd.DataFrame(full_2d, columns=["Next_2D", "Frequency"]))
    else:
        st.warning(f"No direct historical match for 2D '{target_2d}' - falling back to independent composite scores.")

    st.divider()
    st.header("Strong Numbers (Top Composite 4D & Cross-Investment 2D)")
    strong_numbers = generate_strong_numbers(scores)
    st.dataframe(strong_numbers.head(10), use_container_width=True, hide_index=True)

    top_2d = strong_numbers.drop_duplicates("Back_2D").head(5)[["Back_2D", "Total_Score"]]
    st.subheader("Top 5 Recommended 2D (Cross-Investment)")
    st.table(top_2d.reset_index(drop=True))

    st.divider()
    st.caption(
        "Interpretation: The composite score blends Global Frequency (30%), "
        "Recency-Weighted Frequency (30%), and Markov Transition H to H+1 (40%). "
        "This ranking does NOT change the base probability of a truly random RNG - "
        "use it as exploratory reference, not a guaranteed outcome."
    )


if __name__ == "__main__":
    main()
