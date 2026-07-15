import streamlit as st
import pandas as pd
import numpy as np
import re
import gspread
from collections import Counter
from itertools import product
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="GLM 5.2 | Copy-Paste Mode", layout="wide")

SPREADSHEET_ID = "1prsu_8P8rxoKluOdbozwPrCdtJmGd9kBoqzfDnqVlVU"
WORKSHEET_NAME = "DB"
HEADERS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ==========================================
# 1. KONEKSI & DATA WRANGLING
# ==========================================
@st.cache_resource
def get_gsheet_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )
    return gspread.authorize(creds)

@st.cache_data(ttl=600, show_spinner=False)
def load_data():
    try:
        client = get_gsheet_client()
        ws = client.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
        rows = ws.get_all_values()
    except Exception as e:
        raise ValueError(f"Gagal membaca sheet DB: {e}")

    if not rows:
        raise ValueError("Sheet DB kosong.")

    data_rows = rows[1:] if len(rows) > 1 else []
    normalized_rows = [(row + [""] * 7)[:7] for row in data_rows]
    df_sheet = pd.DataFrame(normalized_rows, columns=HEADERS)

    cells = []
    for row in df_sheet.itertuples(index=False):
        for val in row:
            sval = str(val).strip()
            if sval:
                cells.append(sval)

    flat_text = " ".join(cells)
    tokens = re.findall(r"\d{4}", flat_text)

    if len(tokens) < 10:
        digits = re.findall(r"\d", flat_text)
        usable = len(digits) - (len(digits) % 4)
        tokens = ["".join(digits[i:i + 4]) for i in range(0, usable, 4)]

    return np.array(tokens, dtype=str)

# ==========================================
# 2. MESIN LOGIKA (TETAP SAMA - SANGAT AMPUH)
# ==========================================
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
            "Skor_Gabungan": composite
        }).sort_values("Skor_Gabungan", ascending=False).reset_index(drop=True)
        scores.append(df_pos)
    return scores

def full_2d_markov(pairs, target_2d, top_n=6):
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
        s1 = scores[0].set_index("Digit").loc[int(d1), "Skor_Gabungan"]
        s2 = scores[1].set_index("Digit").loc[int(d2), "Skor_Gabungan"]
        s3 = scores[2].set_index("Digit").loc[int(d3), "Skor_Gabungan"]
        s4 = scores[3].set_index("Digit").loc[int(d4), "Skor_Gabungan"]

        combos.append({
            "4D": f"{d1}{d2}{d3}{d4}",
            "3D": f"{d2}{d3}{d4}",
            "2D_Belakang": f"{d3}{d4}",
            "Skor_Total": float(s1 + s2 + s3 + s4)
        })
    return pd.DataFrame(combos).sort_values("Skor_Total", ascending=False).reset_index(drop=True)

def aggregate_digit_strength(scores):
    total = Counter()
    for df_pos in scores:
        for _, row in df_pos.iterrows():
            total[str(int(row["Digit"]))] += float(row["Skor_Gabungan"])
    rows = [{"Digit": k, "Skor_Agregat": v} for k, v in total.items()]
    return pd.DataFrame(rows).sort_values("Skor_Agregat", ascending=False).reset_index(drop=True)

def top_two_digits(scores, pos_index):
    vals = scores[pos_index].head(2)["Digit"].astype(int).astype(str).tolist()
    return f"{vals[0]} & {vals[1]}" if len(vals) > 1 else vals[0]

# ==========================================
# 3. ANTARMUKA COPY-PASTE BLOG
# ==========================================
def main():
    st.title("⚙️ GLM 5.2 (Blog Output Mode)")
    
    try:
        with st.spinner("Menghubungkan ke Database..."):
            history = load_data()
    except Exception as e:
        st.error(f"Kesalahan sistem: {e}")
        return

    st.markdown("### 🎯 Kalibrasi Baseline (H-1)")
    baseline = st.text_input("Masukkan hasil 4D kemarin:", max_chars=4, placeholder="Contoh: 3506")
    
    if st.button("Generate Teks Blog", type="primary"):
        if not baseline.isdigit() or len(baseline) != 4:
            st.error("Input ditolak. Masukkan tepat 4 digit angka.")
            return

        with st.spinner("Memproses Matriks..."):
            pairs = build_pairs(history)
            scores = per_position_scores(history, pairs, baseline)
            strong_numbers = generate_strong_numbers(scores)

            # Tarik daftar list murni tanpa persentase
            target_2d = baseline[2:]
            full_2d = full_2d_markov(pairs, target_2d, top_n=6)
            
            top_4d = strong_numbers["4D"].head(8).tolist()
            top_3d = strong_numbers.drop_duplicates("3D")["3D"].head(8).tolist()
            top_2d = strong_numbers.drop_duplicates("2D_Belakang")["2D_Belakang"].head(8).tolist()
            
            hist_2d_str = ", ".join([row[0] for row in full_2d]) if full_2d else "Belum ada sejarah utuh"

            as_pot = top_two_digits(scores, 0)
            kop_pot = top_two_digits(scores, 1)
            kep_pot = top_two_digits(scores, 2)
            eko_pot = top_two_digits(scores, 3)

            bbfs6 = "".join(aggregate_digit_strength(scores).head(6)["Digit"].astype(str).tolist())

        st.divider()
        st.success("✅ Output siap disalin ke Blog Anda!")

        # ----------------------------------------
        # VISUAL HTML COPY (Menyimpan format bold)
        # ----------------------------------------
        st.subheader("📋 Tampilan Blog (Blok dan Copy Area Ini)")
        
        st.markdown(f"""
**🔍 Analisis Posisi Angka (GLM 5.2):**
* **AS (Ribuan):** {as_pot}
* **KOP (Ratusan):** {kop_pot}
* **KEPALA (Puluhan):** {kep_pot}
* **EKOR (Satuan):** {eko_pot}

**🎯 Matriks Jaring Silang:**
* **Set 4D:** {", ".join(top_4d)}
* **Set 3D:** {", ".join(top_3d)}
* **Set 2D Belakang:** {", ".join(top_2d)}
* **Transisi Sejarah 2D Asli:** {hist_2d_str}

**🔮 Kesimpulan BBFS (6 Digit):**
**[ {bbfs6} ]**
        """)

        st.divider()
        
        # ----------------------------------------
        # TEXT RAW (Kotak Salin Otomatis)
        # ----------------------------------------
        st.subheader("📦 Alternatif: Copy Teks Mentah")
        st.info("Arahkan kursor ke pojok kanan atas kotak di bawah ini, lalu klik ikon 'Copy'.")
        
        raw_text = f"""🔍 Analisis Posisi Angka (GLM 5.2):
- AS (Ribuan): {as_pot}
- KOP (Ratusan): {kop_pot}
- KEPALA (Puluhan): {kep_pot}
- EKOR (Satuan): {eko_pot}

🎯 Matriks Jaring Silang:
- Set 4D: {", ".join(top_4d)}
- Set 3D: {", ".join(top_3d)}
- Set 2D Belakang: {", ".join(top_2d)}
- Transisi Sejarah 2D Asli: {hist_2d_str}

🔮 Kesimpulan BBFS (6 Digit):
[ {bbfs6} ]"""
        
        st.code(raw_text, language="text")

if __name__ == "__main__":
    main()
