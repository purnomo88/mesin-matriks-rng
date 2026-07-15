import streamlit as st
import pandas as pd
import numpy as np
import re
import gspread
from collections import Counter
from itertools import product
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="GLM 7.0 | Gap Saturation Engine", layout="wide")

SPREADSHEET_ID = "1prsu_8P8rxoKluOdbozwPrCdtJmGd9kBoqzfDnqVlVU"
WORKSHEET_NAME = "DB"
HEADERS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ==========================================
# 1. KONEKSI & DATABASE WRANGLING
# ==========================================
@st.cache_resource
def get_gsheet_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )
    return gspread.authorize(creds)

@st.cache_data(ttl=300, show_spinner=False)
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
# 2. RUMUS STATISTIK BARU: GAP SATURATION ENGINE
# ==========================================
def calculate_gap_saturation(data_target):
    # Bersihkan dari nilai None
    clean_data = [x for x in data_target if x is not None]
    total_data = len(clean_data)
    
    if total_data < 10:
        return None, None
    
    # 1. Hitung Jeda (Gap) Per Posisi Digit
    pos_scores = []
    for pos in range(4):
        gap_dict = {str(d): 0 for d in range(10)}
        
        # Scan dari data terbaru (paling bawah/akhir list) mundur ke belakang
        for d in map(str, range(10)):
            gap = 0
            found = False
            for num in reversed(clean_data):
                if num[pos] == d:
                    found = True
                    break
                gap += 1
            # Jika digit tidak pernah muncul sama sekali di sejarah, berikan gap maksimal
            gap_dict[d] = gap if found else total_data
            
        # Normalisasi skor berdasarkan besarnya gap (Semakin lama tidak keluar, skor semakin tinggi)
        df_pos = pd.DataFrame(list(gap_dict.items()), columns=["Digit", "Skor"]).sort_values("Skor", ascending=False).reset_index(drop=True)
        pos_scores.append(df_pos)
        
    # 2. Hitung Jeda (Gap) Blok 2D Belakang Utuh
    gap_2d = {f"{i:02d}": 0 for i in range(100)}
    for d2 in gap_2d.keys():
        gap = 0
        found = False
        for num in reversed(clean_data):
            if num[2:] == d2:
                found = True
                break
            gap += 1
        gap_2d[d2] = gap if found else total_data
        
    df_2d = pd.DataFrame(list(gap_2d.items()), columns=["2D", "Skor"]).sort_values("Skor", ascending=False).reset_index(drop=True)
    
    # Cari 2D yang paling sering keluar secara umum pada hari tersebut (Untuk kolom Sejarah Asli)
    hist_2d_counts = Counter([x[2:] for x in clean_data])
    top_hist_2d = [item[0] for item in hist_2d_counts.most_common(6)]
    
    return pos_scores, df_2d, top_hist_2d

def generate_combinations(pos_scores, df_2d, top_as=2, top_kop=2, top_2d=4):
    as_list = pos_scores[0].head(top_as)["Digit"].tolist()
    kop_list = pos_scores[1].head(top_kop)["Digit"].tolist()
    belakang_list = df_2d.head(top_2d)["2D"].tolist()
    
    combos = []
    for d_as, d_kop, d_2d in product(as_list, kop_list, belakang_list):
        # Skor gabungan dari akumulasi jeda hari
        s_as = pos_scores[0].set_index("Digit").loc[d_as, "Skor"]
        s_kop = pos_scores[1].set_index("Digit").loc[d_kop, "Skor"]
        s_2d = df_2d.set_index("2D").loc[d_2d, "Skor"]
        
        combos.append({
            "4D": f"{d_as}{d_kop}{d_2d}",
            "3D": f"{d_kop}{d_2d}",
            "2D": d_2d,
            "Total_Gap": int(s_as + s_kop + s_2d)
        })
        
    df_c = pd.DataFrame(combos).sort_values("Total_Gap", ascending=False).reset_index(drop=True)
    return df_c

# ==========================================
# 3. ANTARMUKA DASBOR & OUTPUT COPY BLOG
# ==========================================
def main():
    st.title("Sistem ini ditenagai oleh model komputasi matematis kompleks yang mengekstraksi matriks transisi dari database historis berskala besar sebagai referensi probabilitas analitik, bukan sebagai jaminan kepastian mutlak.")
    
    try:
        df_matrix = load_and_clean_matrix()
    except Exception as e:
        st.error(f"Gagal sinkronisasi data: {e}")
        return

    hari_list = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    
    st.markdown("### 🎯 Analisis Jeda Stagnasi Hari")
    target_hari = st.selectbox("Pilih Target Hari Analisis:", hari_list, index=2)
    
    if st.button("Jalankan Gap Saturation Engine", type="primary"):
        data_target = df_matrix[target_hari].tolist()
        
        with st.spinner("Menghitung titik jenuh kejenuhan angka..."):
            pos_scores, df_2d, top_hist_2d = calculate_gap_saturation(data_target)
            
            if pos_scores is None:
                st.error("Data pada hari tersebut terlalu sedikit untuk dianalisis.")
                return
                
            df_combos = generate_combinations(pos_scores, df_2d)
            
            # Format Output 2 Line Bersih
            as_line = f"{pos_scores[0].iloc[0]['Digit']} & {pos_scores[0].iloc[1]['Digit']}"
            kop_line = f"{pos_scores[1].iloc[0]['Digit']} & {front_scores=None or pos_scores[1].iloc[1]['Digit']}"
            
            top_kepala = list(dict.fromkeys([x[0] for x in df_2d["2D"].head(6)]))
            top_ekor = list(dict.fromkeys([x[1] for x in df_2d["2D"].head(6)]))
            kep_line = f"{top_kepala[0]} & {top_kepala[1]}" if len(top_kepala)>1 else top_kepala[0]
            eko_line = f"{top_ekor[0]} & {top_ekor[1]}" if len(top_ekor)>1 else top_ekor[0]
            
            # Ekstraksi Top 4 Line Mutlak
            list_4d = df_combos["4D"].head(4).tolist()
            list_3d = df_combos.drop_duplicates("3D")["3D"].head(4).tolist()
            list_2d = df_combos.drop_duplicates("2D")["2D"].head(4).tolist()
            
            hist_2d_str = ", ".join(top_hist_2d) if top_hist_2d else "-"
            
            # Kompresi BBFS 6 Digit dari penantian terlama
            all_digits = (
                pos_scores[0].head(2)["Digit"].tolist() + 
                pos_scores[1].head(2)["Digit"].tolist() + 
                top_kepala[:2] + top_ekor[:2]
            )
            bbfs_counts = Counter(all_digits)
            bbfs6 = "".join([item[0] for item in bbfs_counts.most_common(6)])
            
        st.divider()
        st.success("✅ Analisis Selesai! Output Siap Disalin.")
        
        raw_text = f"""Analisis Posisi Angka:

AS\t\t: {as_line}
KOP\t\t: {kop_line}
KEPALA\t\t: {kep_line}
EKOR\t\t: {eko_line}

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
