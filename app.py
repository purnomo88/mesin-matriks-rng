import streamlit as st
import pandas as pd
import numpy as np
import re
import gspread
from collections import Counter
from itertools import product
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="GLM 6.0 | Advanced Engine", layout="wide")

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
# 2. MESIN LOGIKA (UPGRADE: JOINT PROBABILITY MATRIX)
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

        # Komposisi Mandiri
        composite = (0.3 * global_freq) + (0.3 * weighted_freq) + (0.4 * markov_freq)
        df_pos = pd.DataFrame({
            "Digit": range(10),
            "Skor_Gabungan": composite
        }).sort_values("Skor_Gabungan", ascending=False).reset_index(drop=True)
        scores.append(df_pos)
    return scores

# FUNGSI BARU: Menghitung Sinergi antar Angka (Membunuh "Frankenstein Effect")
def generate_strong_numbers(scores, history, top_k=(3, 3, 4, 4)):
    top_per_pos = []
    for i, df_pos in enumerate(scores):
        top_per_pos.append(list(df_pos.head(top_k[i])["Digit"].astype(str)))

    # Kalkulasi database masa lalu untuk Joint Probability
    n = len(history)
    hist_2d = Counter([s[2:] for s in history])
    hist_3d = Counter([s[1:] for s in history])

    combos = []
    for d1, d2, d3, d4 in product(*top_per_pos):
        s1 = scores[0].set_index("Digit").loc[int(d1), "Skor_Gabungan"]
        s2 = scores[1].set_index("Digit").loc[int(d2), "Skor_Gabungan"]
        s3 = scores[2].set_index("Digit").loc[int(d3), "Skor_Gabungan"]
        s4 = scores[3].set_index("Digit").loc[int(d4), "Skor_Gabungan"]

        base_score = float(s1 + s2 + s3 + s4)
        
        c_2d = f"{d3}{d4}"
        c_3d = f"{d2}{d3}{d4}"
        
        # Penalti/Bonus Sinergi: Jika 2D/3D ini sering keluar bersama, nilainya meledak.
        synergy_2d = (hist_2d.get(c_2d, 0) / n) * 15  # Bobot pengganda
        synergy_3d = (hist_3d.get(c_3d, 0) / n) * 25
        
        final_score = base_score * (1 + synergy_2d + synergy_3d)

        combos.append({
            "4D": f"{d1}{d2}{d3}{d4}",
            "3D": c_3d,
            "2D_Belakang": c_2d,
            "Skor_Total": final_score
        })
    return pd.DataFrame(combos).sort_values("Skor_Total", ascending=False).reset_index(drop=True)

def full_2d_markov(pairs, target_2d, top_n=6):
    counter = Counter()
    for today, tomorrow in pairs:
        if today[2:] == target_2d:
            counter[tomorrow[2:]] += 1
    return counter.most_common(top_n)

def aggregate_digit_strength(scores):
    total = Counter()
    for df_pos in scores:
        for _, row in df_pos.iterrows():
            total[str(int(row["Digit"]))] += float(row["Skor_Gabungan"])
    rows = [{"Digit": k, "Skor_Agregat": v} for k, v in total.items()]
    return pd.DataFrame(rows).sort_values("Skor_Agregat", ascending=False).reset_index(drop=True)

def extract_pos_info(scores, pos_index):
    vals = scores[pos_index].head(2)["Digit"].astype(int).astype(str).tolist()
    if len(vals) == 1:
        return vals[0], vals[0]
    return f"{vals[0]} & {vals[1]}", vals[0]

# ==========================================
# 3. ANTARMUKA COPY-PASTE BLOG
# ==========================================
def main():
    st.title("Sistem ini ditenagai oleh model komputasi matematis kompleks yang mengekstraksi matriks transisi dari database historis berskala besar sebagai referensi probabilitas analitik, bukan sebagai jaminan kepastian mutlak.")
    
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

        with st.spinner("Menjalankan Matriks Sinergi & Probabilitas Bersama..."):
            pairs = build_pairs(history)
            scores = per_position_scores(history, pairs, baseline)
            
            # MEMANGGIL FUNGSI YANG SUDAH DI-UPGRADE
            strong_numbers = generate_strong_numbers(scores, history)

            target_2d = baseline[2:]
            full_2d = full_2d_markov(pairs, target_2d, top_n=6)
            hist_2d_str = ", ".join([row[0] for row in full_2d]) if full_2d else "Belum ada sejarah utuh"
            
            top_4d_list = strong_numbers["4D"].head(4).tolist()
            top_4d_terkuat = top_4d_list[0] if top_4d_list else "-"
            
            top_3d_list = strong_numbers.drop_duplicates("3D")["3D"].head(4).tolist()
            top_3d_terkuat = top_3d_list[0] if top_3d_list else "-"
            
            top_2d_list = strong_numbers.drop_duplicates("2D_Belakang")["2D_Belakang"].head(4).tolist()
            top_2d_terkuat = top_2d_list[0] if top_2d_list else "-"

            as_pot, as_kuat = extract_pos_info(scores, 0)
            kop_pot, kop_kuat = extract_pos_info(scores, 1)
            kep_pot, kep_kuat = extract_pos_info(scores, 2)
            eko_pot, eko_kuat = extract_pos_info(scores, 3)

            bbfs6 = "".join(aggregate_digit_strength(scores).head(6)["Digit"].astype(str).tolist())

        st.divider()
        st.success("✅ Output siap disalin ke Blog Anda!")
        
        st.info("Arahkan kursor ke pojok kanan atas kotak di bawah ini, lalu klik ikon 'Copy'.")
        
        raw_text = f"""Analisis Posisi Angka:

AS\t\t: {as_pot}\tTERKUAT\t: {as_kuat}
KOP\t\t: {kop_pot}\tTERKUAT\t: {kop_kuat}
KEPALA\t: {kep_pot}\tTERKUAT\t: {kep_kuat}
EKOR\t: {eko_pot}\tTERKUAT\t: {eko_kuat}

Matriks Jaring Silang:

Set 4D Terbaik\t: {", ".join(top_4d_list)}
Set 4D Terkuat\t: {top_4d_terkuat}

Set 3D Terbaik\t: {", ".join(top_3d_list)}
Set 3D Terkuat\t: {top_3d_terkuat}

Set 2D Belakang Terbaik\t: {", ".join(top_2d_list)}
Set 2D Belakang Terkuat\t: {top_2d_terkuat}

Transisi Sejarah 2D Asli\t: {hist_2d_str}

Jaring Silang (6 Digit) BBFS : [ {bbfs6} ]"""
        
        st.code(raw_text, language="text")

if __name__ == "__main__":
    main()
