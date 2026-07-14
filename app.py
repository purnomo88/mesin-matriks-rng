import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from itertools import product
import re

st.set_page_config(page_title="4D Markov Transition Analyzer", layout="wide")

# ---------------------------------------------------------------------------
# STATISTICAL DISCLAIMER
# ---------------------------------------------------------------------------
st.warning(
    "⚠️ Statistical Note: If this data truly originates from an unbiased RNG, "
    "each draw is independent (i.i.d.). Historical 'transition patterns' are "
    "mathematically indistinguishable from noise and carry no predictive power "
    "for the next draw. This tool is provided strictly for exploratory frequency "
    "analysis / entertainment, NOT as a guarantee of future outcomes."
)

# ---------------------------------------------------------------------------
# 1. DATA WRANGLING
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=True)
def load_and_wrangle(sheet_url: str) -> np.ndarray:
    """
    Reads a public Google Sheets CSV export, skips header row,
    flattens all cells, drops NaN/empty entries, and stitches every
    4 consecutive valid digits into a single 4-digit string.
    """
    try:
        df = pd.read_csv(sheet_url, header=None, skiprows=1, dtype=str)
    except Exception as e:
        raise ValueError(f"Failed to fetch/parse Google Sheet CSV: {e}")

    if df.empty:
        raise ValueError("The fetched sheet is empty after skipping the header row.")

    flat = df.values.flatten()

    cleaned = []
    for val in flat:
        if val is None:
            continue
        s = str(val).strip()
        if s == "" or s.lower() == "nan":
            continue
        s = re.sub(r"[^0-9]", "", s)
        if s == "":
            continue
        # If a cell itself contains multiple digits already (e.g. "3506"),
        # split into individual characters to keep the stitching logic
