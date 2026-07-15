import streamlit as st
import pandas as pd
import numpy as np
import re
import gspread
from collections import Counter
from itertools import product
from scipy.stats import chisquare
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Mesin Analisis 4D", layout="wide")

SPREADSHEET_ID = "1prsu_8P8rxoKluOdbozwPrCdtJmGd9kBoqzfDnqVlVU"
WORKSHEET_NAME = "DB"
HEADERS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


@st.cache_resource
def get_gsheet_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )
    return gspread.authorize(creds)


@st.cache_data(ttl=600, show_spinner=True)
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

    if len(tokens) < 10:
        raise ValueError("Data 4 digit valid kurang dari 10 entri.")

    return np.array(tokens, dtype=str), df_sheet


def chi_square_sanity_check(history: np.ndarray):
    n = len(history)
    results = []
    for pos in range(4):
        digits = [int(s[pos]) for s in history]
        obs = np.bincount(digits, minlength=10)
        exp = np.full(10, n / 10)
        chi2, p = chisquare(obs, exp)
        results.append({
            "Posisi": f"Digit {pos + 1}",
            "Chi2": round(chi2, 2),
            "p-value": round(p, 4),
            "Status": "Normal / Acak" if p > 0.05 else "Ada Bias Historis"
        })
    return pd.DataFrame(results)


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
            "Frekuensi_Global": global_freq.round(4),
            "Frekuensi_Recency": weighted_freq.round(4),
            "Frekuensi_Markov": markov_freq.round(4),
            "Skor_Gabungan": composite.round(4)
        }).sort_values("Skor_Gabungan", ascending=False).reset_index(drop=True)

        scores.append(df_pos)

    return scores


def full_2d_markov(pairs, target_2d, top_n=5):
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
            "Skor_Total": round(s1 + s2 + s3 + s4, 4)
        })

    combos = pd.DataFrame(combos).sort_values("Skor_Total", ascending=False).reset_index(drop=True)
    return combos


def top_unique(df, key_col, score_col, n):
    return df.drop_duplicates(key_col).head(n)[[key_col, score_col]].reset_index(drop=True)


def aggregate_digit_strength(scores):
    total = Counter()
    for df_pos in scores:
        for _, row in df_pos.iterrows():
            total[str(int(row["Digit"]))] += float(row["Skor_Gabungan"])
    rows = [{"Digit": k, "Skor_Agregat": round(v, 4)} for k, v in total.items()]
    return pd.DataFrame(rows).sort_values("Skor_Agregat", ascending=False).reset_index(drop=True)


def one_digit_label(scores, pos_index):
    return str(int(scores[pos_index].iloc[0]["Digit"]))


def show_simple_table(title, df, col1, col2=None):
    st.markdown(f"### {title}")
    if col2:
        st.table(df.rename(columns={col1: "Nilai", col2: "Skor"}))
    else:
        st.table(df.rename(columns={col1: "Nilai"}))


def main():
    st.title("Mesin Analisis 4D")
    st.caption("Mode utama: baseline memakai hasil 4D kemarin.")

    st.info(
        "Mesin ini mempertahankan rumus lama Anda, lalu membaca histori langsung dari sheet DB "
        "dan menampilkan hasil dalam Bahasa Indonesia."
    )

    st.sidebar.header("Pengaturan")
    recency_halflife = st.sidebar.slider(
        "Bobot data terbaru (lebih kecil = lebih fokus ke data terbaru)",
        min_value=10,
        max_value=200,
        value=50
    )

    with st.expander("Penjelasan Dashboard", expanded=False):
        st.markdown("""
- **Bobot data terbaru**: mengatur seberapa besar pengaruh data terbaru dibanding data lama.
- **Baseline hasil kemarin**: angka 4 digit kemarin yang dipakai sebagai acuan analisis utama.
- **Skor gabungan**: gabungan dari frekuensi global, bobot data terbaru, dan transisi historis.
- **4D / 3D / 2D skor atas**: ringkasan kombinasi historis dengan skor gabungan tertinggi.
- **As / Kop / Kepala / Ekor**: digit teratas per posisi.
- **BBFS 6 digit**: 6 digit dengan skor agregat tertinggi dari seluruh posisi.
        """)

    try:
        history, df_sheet = load_data()
    except ValueError as ve:
        st.error(f"Kesalahan data: {ve}")
        return
    except Exception as e:
        st.error(f"Kesalahan tak terduga: {e}")
        return

    st.success(f"{len(history)} entri historis 4D berhasil dimuat.")
    st.caption("Sumber data: Google Sheets > sheet DB")

    with st.expander("Preview data DB", expanded=False):
        st.dataframe(df_sheet.tail(20), use_container_width=True)

    with st.expander("Cek distribusi historis", expanded=False):
        st.dataframe(chi_square_sanity_check(history), use_container_width=True)

    pairs = build_pairs(history)

    st.divider()
    st.subheader("Input baseline")
    baseline = st.text_input("Masukkan hasil 4D kemarin:", max_chars=4, placeholder="Contoh: 3506")

    proses = st.button("Proses Analisis", type="primary")

    if not proses:
        return

    baseline = baseline.strip()
    if not baseline.isdigit() or len(baseline) != 4:
        st.error("Input harus tepat 4 digit angka, misalnya 3506.")
        return

    try:
        scores = per_position_scores(history, pairs, baseline, recency_halflife)
        strong_numbers = generate_strong_numbers(scores)
    except Exception as e:
        st.error(f"Kesalahan perhitungan: {e}")
        return

    target_2d = baseline[2:]
    full_2d = full_2d_markov(pairs, target_2d, top_n=4)

    if full_2d:
        df_2d_hist = pd.DataFrame(full_2d, columns=["2D_Belakang", "Frekuensi"])
    else:
        df_2d_hist = top_unique(strong_numbers, "2D_Belakang", "Skor_Total", 4).rename(
            columns={"Skor_Total": "Frekuensi"}
        )

    hasil_4d_4 = strong_numbers.head(4)[["4D", "Skor_Total"]]
    hasil_4d_2 = strong_numbers.head(2)[["4D", "Skor_Total"]]

    hasil_3d_4 = top_unique(
        strong_numbers.rename(columns={"3D": "Nilai_3D"}), "Nilai_3D", "Skor_Total", 4
    )
    hasil_3d_2 = top_unique(
        strong_numbers.rename(columns={"3D": "Nilai_3D"}), "Nilai_3D", "Skor_Total", 2
    )

    hasil_2d_4 = top_unique(strong_numbers, "2D_Belakang", "Skor_Total", 4)
    hasil_2d_2 = top_unique(strong_numbers, "2D_Belakang", "Skor_Total", 2)

    as_top = one_digit_label(scores, 0)
    kop_top = one_digit_label(scores, 1)
    kepala_top = one_digit_label(scores, 2)
    ekor_top = one_digit_label(scores, 3)

    digit_agregat = aggregate_digit_strength(scores)
    bbfs6 = "".join(digit_agregat.head(6)["Digit"].astype(str).tolist())
    digit_umum_teratas = digit_agregat.iloc[0]["Digit"]

    st.divider()
    st.header("Hasil Ringkas")

    col_a, col_b = st.columns(2)

    with col_a:
        show_simple_table("4D Skor Atas (4 baris)", hasil_4d_4, "4D", "Skor_Total")
        show_simple_table("4D Skor Tertinggi (2 baris)", hasil_4d_2, "4D", "Skor_Total")
        show_simple_table("3D Skor Atas (4 baris)", hasil_3d_4, "Nilai_3D", "Skor_Total")
        show_simple_table("3D Skor Tertinggi (2 baris)", hasil_3d_2, "Nilai_3D", "Skor_Total")

    with col_b:
        st.markdown("### 2D Belakang Historis (4 baris)")
        st.table(df_2d_hist)

        show_simple_table("2D Skor Atas (4 baris)", hasil_2d_4, "2D_Belakang", "Skor_Total")
        show_simple_table("2D Skor Tertinggi (2 baris)", hasil_2d_2, "2D_Belakang", "Skor_Total")

    st.divider()
    st.header("Digit Posisi Teratas")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("As Teratas", as_top)
    c2.metric("Kop Teratas", kop_top)
    c3.metric("Kepala Teratas", kepala_top)
    c4.metric("Ekor Teratas", ekor_top)
    c5.metric("Digit Umum Teratas", digit_umum_teratas)
    c6.metric("BBFS 6 Digit", bbfs6)

    st.divider()
    st.header("Skor per Posisi")
    labels = ["As / Ribuan", "Kop / Ratusan", "Kepala / Puluhan", "Ekor / Satuan"]
    cols = st.columns(4)

    for i, (label, df_pos) in enumerate(zip(labels, scores)):
        with cols[i]:
            st.markdown(f"**{label}**")
            st.dataframe(df_pos.head(5), use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "Interpretasi: skor gabungan dibentuk dari Frekuensi Global (30%), "
        "Frekuensi Recency (30%), dan Transisi Markov H ke H+1 (40%)."
    )


if __name__ == "__main__":
    main()
