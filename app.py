import streamlit as st
import pandas as pd
from collections import Counter
import re

st.set_page_config(page_title="Day-Pair Matrix V4.2", layout="wide")

# ==========================================
# 1. AUTO-FIX URL & EXTRACTOR
# ==========================================
def fix_url(url: str) -> str:
    if "/edit" in url: return url.split("/edit")[0] + "/export?format=csv"
    return url

@st.cache_data(ttl=300, show_spinner=False)
def load_and_verify_matrix(sheet_url: str):
    df_raw = pd.read_csv(sheet_url, header=None, skiprows=1)
    
    # Ekstraksi brutal: Cari semua 4 digit angka
    all_numbers = []
    for _, row in df_raw.iterrows():
        row_str = " ".join([str(val) for val in row.values if pd.notna(val)])
        matches = re.findall(r'\b\d{4}\b', row_str)
        all_numbers.extend(matches)
        
    # Validasi Kelipatan 7
    total_found = len(all_numbers)
    if total_found % 7 != 0:
        st.warning(f"⚠️ Peringatan: Data tidak sempurna. Ditemukan {total_found} angka (bukan kelipatan 7). Mungkin ada data hari yang hilang.")
    
    # Distribusi ke Matriks
    weeks = [all_numbers[i:i+7] for i in range(0, total_found - (total_found % 7), 7)]
    df = pd.DataFrame(weeks, columns=["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
    return df

# ==========================================
# 2. MAIN INTERFACE
# ==========================================
def main():
    st.title("⚖️ Day-Pair Matrix V4.2 (Verified)")
    
    with st.sidebar:
        st.header("⚙️ Database Setup")
        raw_url = st.text_input("Google Sheets Link:")
        sheet_url = fix_url(raw_url)

    if not sheet_url:
        st.info("👈 Masukkan link Google Sheets di menu kiri.")
        return

    try:
        df = load_and_verify_matrix(sheet_url)
        st.sidebar.success(f"✅ Data Terbaca: {len(df)} Minggu.")
        
        # FITUR VERIFIKASI VISUAL (PENTING!)
        with st.expander("👁️ Cek Data (Pastikan tabel di bawah ini benar)"):
            st.dataframe(df.head(10)) 
            st.write("Jika tabel di atas berantakan, artinya file CSV sumber Anda memang rusak/tidak beraturan.")

    except Exception as e:
        st.error(f"Gagal membaca data: {e}")
        return

    # LOGIKA ANALISA (Sama seperti V4.1)
    hari_list = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    target_hari = st.selectbox("Pilih Target Hari (Yang Ingin Ditebak):", hari_list, index=2)
    idx_target = hari_list.index(target_hari)
    prev_hari = hari_list[(idx_target - 1) % 7]
    
    baseline = st.text_input(f"Masukkan Hasil {prev_hari} (Baseline):", max_chars=4)

    if st.button("Analisis"):
        b_kep, b_ekor = baseline[2], baseline[3]
        target_2d_utuh, target_k, target_e = [], [], []

        for i in range(len(df)):
            prev_val = df.iloc[i][prev_hari]
            target_val = df.iloc[i][target_hari] if target_hari != "Senin" else (df.iloc[i+1][target_hari] if i+1 < len(df) else None)
            
            if prev_val and target_val:
                if prev_val[2] == b_kep and prev_val[3] == b_ekor:
                    target_2d_utuh.append(target_val[2:])
                if prev_val[2] == b_kep: target_k.append(target_val[2])
                if prev_val[3] == b_ekor: target_e.append(target_val[3])

        # OUTPUT
        if target_2d_utuh:
            st.success(f"2D Utuh: {Counter(target_2d_utuh).most_common(3)}")
        st.info(f"Kepala Kuat: {Counter(target_k).most_common(2)} | Ekor Kuat: {Counter(target_e).most_common(2)}")

if __name__ == "__main__":
    main()
