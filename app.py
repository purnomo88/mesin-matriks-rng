import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re

st.set_page_config(page_title="4D Markov Transition Engine", layout="wide")

# ==========================================
# 1. AUTO-FIX URL ENGINE (Self-Healing Link)
# ==========================================
def fix_url(url: str) -> str:
    if not url:
        return ""
    if "/edit" in url:
        return url.split("/edit")[0] + "/export?format=csv"
    if not url.endswith("/export?format=csv"):
        if "?" in url:
            return url.split("?")[0] + "/export?format=csv"
        else:
            return url + "/export?format=csv"
    return url

# ==========================================
# 2. DATA WRANGLING (Claude's Logic)
# ==========================================
@st.cache_data(ttl=600, show_spinner=False)
def load_and_wrangle(sheet_url: str) -> np.ndarray:
    try:
        df = pd.read_csv(sheet_url, header=None, skiprows=1, dtype=str)
    except Exception as e:
        raise ValueError(f"Failed to fetch CSV. Ensure the link is public. Error: {e}")

    flat = df.values.flatten()
    cleaned = []
    
    for val in flat:
        if pd.isna(val): 
            continue
        s = str(val).strip()
        if s == "" or s.lower() == "nan": 
            continue
        # Strip all non-numeric characters (Bulletproof parsing)
        s = re.sub(r"[^0-9]", "", s)
        if s == "": 
            continue
        for ch in s:
            cleaned.append(ch)

    usable_len = len(cleaned) - (len(cleaned) % 4)
    trimmed = cleaned[:usable_len]
    # Stitch back into 4-digit strings
    stitched = ["".join(trimmed[i:i + 4]) for i in range(0, usable_len, 4)]
    return np.array(stitched, dtype=str)

# ==========================================
# 3. MARKOV CHAIN ENGINE (H+1 Transition)
# ==========================================
def build_pairs(history):
    return list(zip(history[:-1], history[1:]))

def get_2d_transitions(pairs, target_2d):
    counter = Counter()
    for today, tomorrow in pairs:
        if len(today) == 4 and today[2:] == target_2d:
            if len(tomorrow) == 4:
                counter[tomorrow[2:]] += 1
    return counter.most_common(3)

def get_position_transitions(pairs, baseline):
    counters = [Counter() for _ in range(4)]
    for today, tomorrow in pairs:
        if len(today) != 4 or len(tomorrow) != 4: 
            continue
        for pos in range(4):
            if today[pos] == baseline[pos]:
                counters[pos][tomorrow[pos]] += 1
    return counters

# ==========================================
# 4. MAIN USER INTERFACE
# ==========================================
def main():
    st.title("🔬 4D Markov Transition Engine (Auto-Link)")
    st.markdown("Historical frequency analyzer with self-healing database connection.")

    with st.sidebar:
        st.header("⚙️ Database Setup")
        raw_url = st.text_input("Google Sheets Link:", placeholder="Paste your link here...")
        sheet_url = fix_url(raw_url)

    if not sheet_url:
        st.info("👈 Please paste your Google Sheets link in the sidebar to begin.")
        return

    with st.spinner("Fetching and wrangling database..."):
        try:
            history = load_and_wrangle(sheet_url)
            st.sidebar.success(f"✅ Connection Stable. {len(history)} rows loaded.")
        except Exception as e:
            st.sidebar.error(str(e))
            return

    pairs = build_pairs(history)

    st.subheader("Today's Baseline Input")
    baseline = st.text_input("Enter Today's 4D Result:", max_chars=4, placeholder="e.g. 9392")
    
    if st.button("Calculate H+1 Transitions", type="primary"):
        if len(baseline) != 4 or not baseline.isdigit():
            st.error("Invalid input. Please enter exactly 4 digits.")
            return
            
        target_2d = baseline[2:]
        
        st.divider()
        st.header("🎯 Absolute Target (2D Unit Transition)")
        full_2d = get_2d_transitions(pairs, target_2d)
        
        if full_2d:
            st.success(f"Historical matches found for 2D tail '{target_2d}'.")
            df_2d = pd.DataFrame(full_2d, columns=["Next_2D_Tail", "Frequency"])
            df_2d.index += 1
            st.table(df_2d)
        else:
            st.warning(f"No exact historical match for 2D tail '{target_2d}'. Generating cross-investment matrix.")
            
        st.divider()
        st.header("📊 Independent Position Extraction")
        pos_counters = get_position_transitions(pairs, baseline)
        labels = ["Digit 1 (As)", "Digit 2 (Kop)", "Digit 3 (Kepala)", "Digit 4 (Ekor)"]
        
        cols = st.columns(4)
        top_d3, top_d4 = [], []
        
        for i, (label, counter) in enumerate(zip(labels, pos_counters)):
            limit = 3 if i >= 2 else 2 # Top 2 for D1/D2, Top 3 for D3/D4
            top_n = counter.most_common(limit)
            
            if i == 2: top_d3 = top_n
            if i == 3: top_d4 = top_n
            
            with cols[i]:
                st.markdown(f"**{label}**")
                if top_n:
                    st.table(pd.DataFrame(top_n, columns=["Digit", "Freq"]))
                else:
                    st.write("No data.")

        if not full_2d and top_d3 and top_d4:
            st.divider()
            st.header("🧩 Cross-Investment 2D Matrix")
            cross = []
            for d3, c3 in top_d3:
                for d4, c4 in top_d4:
                    cross.append({
                        "2D_Combo": f"{d3}{d4}", 
                        "Digit_3": d3, "Freq_3": c3,
                        "Digit_4": d4, "Freq_4": c4,
                        "Combined_Score": c3 + c4
                    })
            cross.sort(key=lambda x: x["Combined_Score"], reverse=True)
            df_cross = pd.DataFrame(cross)
            df_cross.index += 1
            st.dataframe(df_cross, use_container_width=True)

if __name__ == "__main__":
    main()
