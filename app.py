import streamlit as st
import pandas as pd
import numpy as np
import re
import gspread
from collections import Counter
from itertools import product
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="GLM 6.0 | Calibrated Engine", layout="wide")

SPREADSHEET_ID = "1prsu_8P8rxoKluOdbozwPrCdtJmGd9kBoqzfDnqVlVU"
WORKSHEET_NAME = "DB"
HEADERS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ==========================================
# 1. KONEKSI & WRANGLING DATA
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
# 2. MESIN MATEMATIKA (KALIBRASI DINAMIS TREN)
# ==========================================
def hybrid_engine_order2(data_prev, data_target, baseline, recency_limit=60):
    pairs = []
    for i in range(len(data_prev)):
        today = data_prev[i]
        tomorrow = data_target[i]
        if today and tomorrow:
            pairs.append((today, tomorrow))
            
    recent_pairs = pairs[-recency_limit:] if len(pairs) > recency_limit else pairs
    
    # A. Kalibrasi Posisi Depan (AS & KOP)
    front_scores = []
    for pos in range(2):
        b_digit = baseline[pos]
        m_counter = Counter([tom[pos] for tod, tom in pairs if tod[pos] == b_digit])
        r_counter = Counter([tom[pos] for tod, tom in recent_pairs if tod[pos] == b_digit])
        
        pos_scores = {str(d): 0.1 for d in range(10)}
        
        tot_m = sum(m_counter.values())
        if tot_m > 0:
            for d, c in m_counter.items(): pos_scores[d] += (c / tot_m) * 0.4 # Markov 40%
                
        tot_r = sum(r_counter.values())
        if tot_r > 0:
            for d, c in r_counter.items(): pos_scores[d] += (c / tot_r) * 0.4 # Recency dinaikkan ke 40% (Lebih adaptif)
            
        df_pos = pd.DataFrame(list(pos_scores.items()), columns=["Digit", "Skor"]).sort_values("Skor", ascending=False)
        front_scores.append(df_pos)
        
    # B. Kalibrasi Blok 2D Belakang (Order-2)
    b_2d = baseline[2:]
    m_2d_counter = Counter([tom[2:] for tod, tom in pairs if tod[2:] == b_2d])
    r_2d_counter = Counter([tom[2:] for tod, tom in recent_pairs if tod[2:] == b_2d])
    g_2d_counter = Counter([tom[2:] for tod, tom in pairs])
    
    tot_g_2d = sum(g_2d_counter.values())
    tot_m_2d = sum(m_2d_counter.values())
    tot_r_2d = sum(r_2d_counter.values())
    
    scores_2d = {}
    for i in range(100):
        d2_str = f"{i:02d}"
        s_g = (g_2d_counter[d2_str] / tot_g_2d) * 0.1 if tot_g_2d else 0.01 # Global diturunkan ke 10%
        s_m = (m_2d_counter[d2_str] / tot_m_2d) * 0.4 if tot_m_2d else 0 # Markov 40%
        s_r = (r_2d_counter[d2_str] / tot_r_2d) * 0.4 if tot_r_2d else 0 # Recency 40%
        scores_2d[d2_str] = s_g + s_m + s_r + 0.01
        
    df_2d = pd.DataFrame(list(scores_2d.items()), columns=["2D", "Skor"]).sort_values("Skor", ascending=False)
    
    hist_2d_matches = [item[0] for item in m_2d_counter.most_common(6)]
    
    # C. Twin Filter
    twin_count = sum(1 for _, tom in recent_pairs if tom[0] == tom[1])
    twin_rate = twin_count / len(recent_pairs) if recent_pairs else 0
    twin_multiplier = 1.4 if twin_rate > 0.15 else 0.5 
    
    return front_scores, df_2d, hist_2d_matches, twin_multiplier

def generate_combinations(front_scores, df_2d, twin_multiplier, top_as=3, top_kop=3, top_2d=6):
    as_list = front_scores[0].head(top_as).values.tolist()
    kop_list = front_scores[1].head(top_kop).values.tolist()
    belakang_list = df_2d.head(top_2d).values.tolist()
    
    combos = []
    for (d_as, s_as), (d_kop, s_kop), (d_2d, s_2d) in product(as_list, kop_list, belakang_list):
        base_score = s_as + s_kop + s_2d
        if d_as == d_kop:
            base_score *= twin_multiplier
            
        combos.append({
            "4D": f"{d_as}{d_kop}{d_2d}",
            "3D": f"{d_kop}{d_2d}",
            "2D": d_2d,
            "Total": base_score
        })
        
    df_c = pd.DataFrame(combos).sort_values("Total", ascending=False).reset_index(drop=True)
    return df_c

# ==========================================
# 3. ANTARMUKA DASBOR
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
        
    if st.button("Jalankan Pemurnian GLM 6.0", type="primary"):
        if len(baseline) != 4 or not baseline.isdigit():
            st.error("Input eror. Masukkan tepat 4 digit.")
            return
            
        data_prev = df_matrix[prev_hari].tolist()
        data_target = df_matrix[target_hari].tolist()
        
        if target_hari == "Senin":
            data_target = data_target[1:] + [""]
            data_prev = data_prev[:len(data_target)]
            
        with st.spinner("Memproses Analisis Probabilitas Terkalibrasi..."):
            front_scores, df_2d, hist_2d_str_list, twin_mult = hybrid_engine_order2(data_prev, data_target, baseline)
            df_combos = generate_combinations(front_scores, df_2d, twin_mult)
            
            as_line = f"{front_scores[0].iloc[0]['Digit']} & {front_scores[0].iloc[1]['Digit']}"
            kop_line = f"{front_scores[1].iloc[0]['Digit']} & {front_scores[1].iloc[1]['Digit']}"
            
            top_kepala = list(dict.fromkeys([x[0] for x in df_2d["2D"].head(6)]))
            top_ekor = list(dict.fromkeys([x[1] for x in df_2d["2D"].head(6)]))
            kep_line = f"{top_kepala[0]} & {top_kepala[1]}" if len(top_kepala)>1 else top_kepala[0]
            eko_line = f"{top_ekor[0]} & {top_ekor[1]}" if len(top_ekor)>1 else top_ekor[0]
            
            list_4d = df_combos["4D"].head(4).tolist()
            list_3d = df_combos.drop_duplicates("3D")["3D"].head(4).tolist()
            list_2d = df_combos.drop_duplicates("2D")["2D"].head(4).tolist()
            
            hist_2d_str = ", ".join(hist_2d_str_list) if hist_2d_str_list else "Belum ada sejarah utuh"
            
            all_digits = (
                front_scores[0].head(2)["Digit"].tolist() + 
                front_scores[1].head(2)["Digit"].tolist() + 
                top_kepala[:2] + top_ekor[:2]
            )
            bbfs_counts = Counter(all_digits)
            bbfs6 = "".join([item[0] for item in bbfs_counts.most_common(6)])
        
        st.divider()
        st.success("✅ Kalibrasi Sukses! Output Siap Disalin.")
        
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
