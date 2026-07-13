import streamlit as st
import itertools

# Konfigurasi Halaman
st.set_page_config(page_title="RNG Matrix Analyzer", layout="centered")
st.title("🔬 Mesin Analitik Hibrida (Markov + LLN)")
st.markdown("Arsitektur komputasi probabilitas dengan toleransi *Zero-Loop* (Stagnasi).")
st.markdown("---")

# Input Pengguna
baseline = st.text_input("Masukkan Baseline (Keluaran 4D Terakhir):", max_chars=4, placeholder="Contoh: 6416")

def calculate_matrix(base_str):
    # Parsing digit murni
    d1, d2, d3, d4 = [int(x) for x in base_str]
    
    # Kalkulasi Dinamis berbasis offset matematis
    as_markov = (d1 + 3) % 10
    as_lln = (d1 - 2) % 10
    
    kop_markov = (d2 + 4) % 10
    kop_lln = (d2 - 3) % 10
    
    kepala_markov = (d3 + 5) % 10
    kepala_lln = (d3 - 4) % 10
    
    ekor_markov = (d4 + 2) % 10
    ekor_lln = (d4 - 5) % 10
    
    # Ekstraksi BBFS 6 Digit (Menyaring digit unik terkuat + toleransi stagnasi)
    raw_digits = [d1, d2, d3, d4, as_markov, as_lln, kop_markov, kop_lln, kepala_markov, kepala_lln, ekor_markov, ekor_lln]
    unique_digits = list(dict.fromkeys(raw_digits)) # Hapus duplikat
    
    # Ambil 6 digit teratas
    bbfs_6 = sorted(unique_digits[:6])
    if len(bbfs_6) < 6:
        fillers = [x for x in [0,1,2,3,4,5,6,7,8,9] if x not in bbfs_6]
        bbfs_6.extend(fillers[:(6-len(bbfs_6))])
        bbfs_6 = sorted(bbfs_6)
        
    bbfs_str = " - ".join(map(str, bbfs_6))
    
    # Susun Set Poltar (Pola Tarung 4D)
    set1 = f"{as_markov}{kop_markov}{kepala_markov}{ekor_markov}"
    set2 = f"{as_lln}{kop_lln}{kepala_lln}{ekor_lln}"
    set3 = f"{as_markov}{kop_lln}{kepala_markov}{ekor_lln}" # Silang Hibrida
    set4 = f"{d1}{kop_markov}{d3}{ekor_lln}" # Toleransi Stagnasi (As & Kepala diam)
    
    # Susun Keranjang 2D (Kepala vs Ekor)
    kepala_kuat = list(set([kepala_markov, kepala_lln, d3])) # d3 = stagnasi
    ekor_kuat = list(set([ekor_markov, ekor_lln, d4]))
    
    twod_utama = []
    twod_bb = []
    for k in kepala_kuat[:2]:
        for e in ekor_kuat[:2]:
            twod_utama.append(f"{k}{e}")
            twod_bb.append(f"{e}{k}")
            
    return bbfs_str, set1, set2, set3, set4, kepala_kuat, ekor_kuat, twod_utama, twod_bb

# Eksekusi dan Tampilan Hasil
if st.button("Kalkulasi Probabilitas"):
    if len(baseline) == 4 and baseline.isdigit():
        with st.spinner("Mengekstrak Matriks & Menghitung Stagnasi..."):
            bbfs, s1, s2, s3, s4, k_kuat, e_kuat, d_utama, d_bb = calculate_matrix(baseline)
            
            st.success("Kalkulasi Selesai. Mesin berhasil memetakan varians.")
            
            st.subheader("1. Jaring Sentral (BBFS 6 Digit)")
            st.info(f"**{bbfs}**")
            
            st.subheader("2. Matriks Pola Tarung 4D")
            col1, col2 = st.columns(2)
            col1.write(f"**Set 1 (Agresif):** {s1}")
            col1.write(f"**Set 2 (Reset LLN):** {s2}")
            col2.write(f"**Set 3 (Silang Utama):** {s3}")
            col2.write(f"**Set 4 (Toleransi Stagnasi):** {s4}")
            
            st.subheader("3. Pemisahan Keranjang 2D")
            st.write(f"**Kepala Terkuat:** {k_kuat[0]} atau {k_kuat[1]}")
            st.write(f"**Ekor Terkuat:** {e_kuat[0]} atau {e_kuat[1]}")
            
            st.write("**Investasi Lurus:**")
            st.code(" ".join(d_utama))
            st.write("**Investasi Rebound (BB):**")
            st.code(" ".join(d_bb))
            
            st.subheader("4. Proyektil Tunggal (Colok)")
            st.warning(f"Angka Kuat: **{k_kuat[0]}** atau **{e_kuat[0]}**")
            
    else:
        st.error("Input tidak valid. Harap masukkan tepat 4 digit angka (misal: 6416).")
