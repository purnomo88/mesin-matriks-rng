import streamlit as st
import pandas as pd
from collections import Counter

st.set_page_config(page_title="RNG Matrix V2.0", layout="centered")
st.title("🔬 Mesin Analitik Transisi (Live Database)")
st.markdown("Menganalisis probabilitas berdasarkan frekuensi empiris dari Google Sheets.")

@st.cache_data(ttl=60) # Refresh data otomatis setiap 60 detik
def load_live_data():
    try:
        # MASUKKAN LINK GOOGLE SHEETS ANDA DI BAWAH INI (Di antara tanda kutip)
        g_sheet_url = "https://docs.google.com/spreadsheets/d/1dGmyGFzkmaxZWb6BZGsInS0OZZdxCweUEEEGabubaa8/edit?usp=sharing"
        
        # Mesin otomatis mengubah link share menjadi link sedot CSV murni
        if "/edit" in g_sheet_url:
            csv_export_url = g_sheet_url.split("/edit")[0] + "/export?format=csv"
        else:
            csv_export_url = g_sheet_url

        # Membaca data langsung dari awan (melewati baris nama hari)
        df = pd.read_csv(csv_export_url, header=None, skiprows=1)
        
        raw_values = df.values.flatten()
        clean_values = [v for v in raw_values if pd.notna(v)]
        
        digit_list = [str(int(v)) for v in clean_values if str(v).replace('.0','').isdigit()]
        
        history_4d = []
        for i in range(0, len(digit_list), 4):
            if i + 3 < len(digit_list):
                angka_utuh = f"{digit_list[i]}{digit_list[i+1]}{digit_list[i+2]}{digit_list[i+3]}"
                history_4d.append(angka_utuh)
                
        return history_4d
    
    except Exception as e:
        st.error(f"Gagal menarik data dari Google Sheets. Pastikan link sudah berstatus 'Anyone with the link'. Error Detail: {e}")
        return None

data_historis = load_live_data()

baseline = st.text_input("Masukkan Baseline 4D Hari Ini:", max_chars=4, placeholder="Contoh: 9392")

if st.button("Kalkulasi Rekam Jejak"):
    if data_historis is None:
        st.warning("Menunggu koneksi database yang valid...")
    elif len(baseline) == 4 and baseline.isdigit():
        with st.spinner("Menyedot data dari Google Sheets dan memindai kronologi..."):
            
            b_as, b_kop, b_kepala, b_ekor = baseline[0], baseline[1], baseline[2], baseline[3]
            
            next_kepala, next_ekor, next_2d = [], [], []
            next_as, next_kop = [], []
            
            for i in range(len(data_historis) - 1):
                hari_ini = data_historis[i]
                besoknya = data_historis[i+1]
                
                h_as, h_kop, h_kepala, h_ekor = hari_ini[0], hari_ini[1], hari_ini[2], hari_ini[3]
                
                if h_as == b_as: next_as.append(besoknya[0])
                if h_kop == b_kop: next_kop.append(besoknya[1])
                if h_kepala == b_kepala: next_kepala.append(besoknya[2])
                if h_ekor == b_ekor: next_ekor.append(besoknya[3])
                
                if h_kepala == b_kepala and h_ekor == b_ekor:
                    next_2d.append(f"{besoknya[2]}{besoknya[3]}")
                    
            def get_top_digits(arr, limit=2):
                if not arr: return ["X"] * limit
                most_common = Counter(arr).most_common(limit)
                return [item[0] for item in most_common] + ["X"] * (limit - len(most_common))

            top_kepala = get_top_digits(next_kepala, 3)
            top_ekor = get_top_digits(next_ekor, 3)
            top_as = get_top_digits(next_as, 2)
            top_kop = get_top_digits(next_kop, 2)
            
            st.success(f"Analisis Selesai. Total Riwayat Terdeteksi: {len(data_historis)} Baris 4D.")
            
            st.subheader("🎯 Target Mutlak (Positif Kuat 2D)")
            if next_2d:
                top_2d_utuh = [item[0] for item in Counter(next_2d).most_common(3)]
                st.write(f"Riwayat menemukan angka **{b_kepala}{b_ekor}** di posisi belakang.")
                st.info(f"Paling sering disusul oleh 2D: **{', '.join(top_2d_utuh)}**")
            else:
                st.warning(f"Belum ada riwayat kombinasi 2D '{b_kepala}{b_ekor}' yang terekam. Menggunakan ekstraksi silang terpisah (Kepala vs Ekor):")
                silang_2d = []
                for k in top_kepala[:2]:
                    for e in top_ekor[:2]:
                        if k != "X" and e != "X":
                            silang_2d.append(f"{k}{e}")
                st.info(f"Investasi Silang Berdasarkan Frekuensi: **{', '.join(silang_2d)}**")
            
            st.subheader("📊 Ekstraksi Posisi Mandiri")
            st.write(f"Kepala Kuat (Pantulan dari {b_kepala}): **{top_kepala[0]}, {top_kepala[1]}, {top_kepala[2]}**")
            st.write(f"Ekor Kuat (Pantulan dari {b_ekor}): **{top_ekor[0]}, {top_ekor[1]}, {top_ekor[2]}**")
            
            st.subheader("🛡️ Referensi Poltar 4D Terkuat")
            st.code(f"{top_as[0]}{top_kop[0]}{top_kepala[0]}{top_ekor[0]} / {top_as[1]}{top_kop[1]}{top_kepala[1]}{top_ekor[1]}")
            
    else:
        st.error("Input tidak valid. Harap masukkan tepat 4 digit.")
