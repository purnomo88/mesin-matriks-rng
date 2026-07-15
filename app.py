import streamlit as st
import pandas as pd
import numpy as np
import re
import gspread
from collections import Counter
from itertools import product
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="GLM 5.2 | Pure Day-Pair Engine", layout="wide")

SPREADSHEET_ID = "1prsu_8P8rxoKluOdbozwPrCdtJmGd9kBoqzfDnqVlVU"
WORKSHEET_NAME = "DB"
HEADERS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ==========================================
# 1. KONEKSI & WRANGLING DATA (STRUKTUR HARI TETAP UTUH)
# ==========================================
@st.cache_resource
def get_gsheet_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )
    return gspread.authorize(creds)

@st.cache_data(ttl=600, show_spinner=False)
def load_and_clean_matrix():
    try:
        client = get_gsheet_client()
        ws = client.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
        rows = ws.get_all_values()
    except Exception as e:
        raise ValueError(f"Gagal membaca data: {e}")
    
    if not rows:
        raise ValueError("Database kosong.")
        
    data_rows = rows[1:] if len(rows) > 1 else []
    normalized_rows = [(row + [""] * 7)[:7] for row in data_rows]
    df = pd.DataFrame(normalized_rows, columns=HEADERS)
    
    def clean_4d(val):
        if pd.isna(val): return None
        s = re.sub(r"[^0-9]", "", str(val).strip())
        return s if len(s) == 4 else None

    for col in df.columns:
        df[col] = df[col].apply(clean_4d)
        
    return df

# ==========================================
# 2. CORE MATHEMATICS (RESTORED TO V4.2 HYBRID)
# ==========================================
def per_position_hybrid(data_prev, data_target, baseline, target_hari, recency_limit=60):
    # Filter data valid berpasangan
    pairs = []
    for i in range(len(data_prev)):
        today = data_prev[i]
        tomorrow = data_target[i]
        if target_hari == "Senin" and i + 1 < len(data_prev):
            tomorrow = data_target[i+1]
        if today and tomorrow:
            pairs.append((today, tomorrow))
            
    # Ambil tren terbaru (Recency Window) untuk mengunci pergeseran aktif
    recent_pairs = pairs[-recency_limit:] if len(pairs) > recency_limit else pairs
    
    scores = []
    # Jalankan kalkulasi per posisi (As, Kop, Kepala, Ekor)
    for pos in range(4):
        b_digit = baseline[pos]
        
        # 1. Markov Bersyarat Hari Spesifik
        markov_counter = Counter()
        for today, tomorrow in pairs:
            if today[pos] == b_digit:
                markov_counter[tomorrow[pos]] += 1
                
        # 2. Recency Counter (Tren 60 minggu terakhir pada hari spesifik)
        recency_counter = Counter()
        for today, tomorrow in recent_pairs:
            if today[pos] == b_digit:
                recency_counter[tomorrow[pos]] += 1
                
        # 3. Stagnation Offset (Toleransi tarikan matematika +3 / -2 / Tetap)
        b_int = int(b_digit)
        offsets = [str((b_int + 3) % 10), str((b_int - 2) % 10), b_digit]
        
        # Penggabungan Skor tanpa sistem Multiplier Sinergi palsu
        pos_scores = Counter()
        # Isi semua digit 0-9 dengan bobot dasar yang merata
        for d in map(str, range(10)):
            pos_scores[d] = 0.1
            
        # Tambahkan bobot Markov (Kekuatan Sejarah)
        total_m = sum(markov_counter.values())
        if total_m > 0:
            for d, c in markov_counter.items():
                pos_scores[d] += (c / total_m) * 0.5
                
        # Tambahkan bobot Recency (Kekuatan Tren Baru)
        total_r = sum(recency_counter.values())
        if total_r > 0:
            for d, c in recency_counter.items():
                pos_scores[d] += (c / total_r) * 0.3
                
        # Tambahkan bobot Toleransi Stagnasi
        for d in offsets:
            pos_scores[d] += 0.1
            
        df_pos = pd.DataFrame(pos_scores.most_common(10), columns=["Digit", "Skor"])
        scores.append(df_pos)
        
    return scores, pairs

def generate_output_matrix(scores, pairs, baseline, top_k=(2, 2, 3, 3)):
    top_per_pos = [list(scores[i].head(top_k[i])["Digit"].tolist()) for i in range(4)]
    
    # 2D Sejarah Utuh (Markov Lurus Bersyarat)
    b_2d = baseline[2:]
    hist_2d_counter = Counter()
    for today, tomorrow in pairs:
        if today[2:] == b_2d:
            hist_2d_counter[tomorrow[2:]] += 1
    full_2d_matches = [item[0] for item in hist_2d_counter.most_common(6)]
    
    # Jaring Kombinasi
    combos = []
    for d1, d2, d3, d4 in product(*top_per_pos):
        s1 = scores[0].set_index("Digit").loc[d1, "Skor"]
        s2 = scores[1].set_index("Digit").loc[d2, "Skor"]
        s3 = scores[2].set_index("Digit").loc[d3, "Skor"]
        s4 = scores[3].set_index("Digit").loc[d4, "Skor"]
        
        combos.append({
            "4D": f"{d1}{d2}{d3}{d4}",
            "3D": f"{d2}{d3}{d4}",
            "2D": f"{d3}{d4}",
            "Total": s1 + s2 + s3 + s4
        })
        
    df_c = pd.DataFrame(combos).sort_values("Total", ascending=False).reset_index(drop=True)
    return df_c, full_2d_matches

# ==========================================
# 3. ANTARMUKA DASBOR (COMPACT & PURE TEXT)
# ==========================================
def main():
    st.title("Sistem ini ditenagai oleh model komputasi matematis kompleks yang mengekstraksi matriks transisi dari database historis berskala besar sebagai referensi probabilitas analitik, bukan sebagai jaminan kepastian mutlak.")
    
    try:
        df_matrix = load_and_clean_matrix()
    except Exception as e:
        st.error(f"Gagal sinkronisasi data: {e}")
        return

    hari_list = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    
    st.markdown("### 🎯 Analisis Konfigurasi Hari")
    colA, colB = st.columns(2)
    with colA:
        target_hari = st.selectbox("Pilih Target Hari:", hari_list, index=2)
    
    idx_target = hari_list.index(target_hari)
    prev_hari = hari_list[(idx_target - 1) % 7]
    
    with colB:
        baseline = st.text_input(f"Masukkan Hasil {prev_hari} (Baseline H-1):", max_chars=4)
        
    if st.button("Jalankan Pemurnian Matriks Hari", type="primary"):
        if len(baseline) != 4 or not baseline.isdigit():
            st.error("Input eror. Masukkan tepat 4 digit.")
            return
            
        data_prev = df_matrix[prev_hari].tolist()
        data_target = df_matrix[target_hari].tolist()
        
        # Eksekusi Rumus Murni Kembali
        scores, pairs = per_position_hybrid(data_prev, data_target, baseline, target_hari)
        df_combos, full_2d_matches = generate_output_matrix(scores, pairs, baseline)
        
        # Ekstraksi Data Posisi
        def get_line(df_pos):
            top2 = df_pos.head(2)["Digit"].tolist()
            return f"{top2[0]} & {top2[1]}", top2[0]
            
        as_pot, as_kuat = get_line(scores[0])
        kop_pot, kop_kuat = get_line(scores[1])
        kep_pot, kep_kuat = get_line(scores[2])
        eko_pot, eko_kuat = get_line(scores[3])
        
        # Ekstraksi BBFS
        all_digits = []
        for df_pos in scores:
            all_digits.extend(df_pos.head(3)["Digit"].tolist())
        bbfs_counts = Counter(all_digits)
        bbfs6 = "".join([item[0] for item in bbfs_counts.most_common(6)])
        
        # Ekstraksi Top Line
        list_4d = df_combos["4D"].head(4).tolist()
        list_3d = df_combos.drop_duplicates("3D")["3D"].head(4).tolist()
        list_2d = df_combos.drop_duplicates("2D")["2D"].head(4).tolist()
        
        hist_2d_str = ", ".join(full_2d_matches) if full_2d_matches else "Belum ada sejarah utuh"
        
        st.divider()
        st.success("✅ Output Murni Siap Disalin!")
        st.info("Klik ikon 'Copy' di pojok kanan atas boks di bawah ini untuk langsung ditempel ke Blog.")
        
        raw_text = f"""Analisis Posisi Angka:

AS\t\t: {as_pot}\tTERKUAT\t: {as_kuat}
KOP\t\t: {kop_pot}\tTERKUAT\t: {kop_kuat}
KEPALA\t: {kep_pot}\tTERKUAT\t: {kep_kuat}
EKOR\t: {eko_pot}\tTERKUAT\t: {eko_kuat}

Matriks Jaring Silang:

Set 4D Terbaik\t: {", ".join(list_4d)}
Set 4D Terkuat\t: {list_4d[0]}

Set 3D Terbaik\t: {", ".join(list_3d)}
Set 3D Terkuat\t: {list_3d[0]}

Set 2D Belakang Terbaik\t: {", ".join(list_2d)}
Set 2D Belakang Terkuat\t: {list_2d[0]}

Transisi Sejarah 2D Asli\t: {hist_2d_str}

Jaring Silang (6 Digit) BBFS : [ {bbfs6} ]"""

        st.code(raw_text, language="text")

if __name__ == "__main__":
    main()
