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

    return np.array(tokens, dtype=str)


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
            "Frekuensi_Global": global_freq,
            "Frekuensi_Recency": weighted_freq,
            "Frekuensi_Markov": markov_freq,
            "Skor_Gabungan": composite
        }).sort_values("Skor_Gabungan", ascending=False).reset_index(drop=True)

        scores.append(df_pos)

    return scores


def full_2d_markov(pairs, target_2d, top_n=4):
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


def top_unique(df, key_col, score_col, n):
    return df.drop_duplicates(key_col).head(n)[[key_col, score_col]].reset_index(drop=True)


def aggregate_digit_strength(scores):
    total = Counter()
    for df_pos in scores:
        for _, row in df_pos.iterrows():
            total[str(int(row["Digit"]))] += float(row["Skor_Gabungan"])
    rows = [{"Digit": k, "Skor_Agregat": v} for k, v in total.items()]
    return pd.DataFrame(rows).sort_values("Skor_Agregat", ascending=False).reset_index(drop=True)


def top_two_digits(scores, pos_index):
    vals = scores[pos_index].head(2)["Digit"].astype(int).astype(str).tolist()
    if len(vals) == 1:
        return vals[0]
    return f"{vals[0]} dan {vals[1]}"


def to_percent_series(df, score_col):
    df = df.copy()
    max_score = df[score_col].max() if not df.empty else 0
    if max_score <= 0:
        df["Persen"] = 0
    else:
        df["Persen"] = (df[score_col] / max_score * 100).round(1)
    return df


def render_lines(title, df, key_col, percent_col="Persen"):
    st.markdown(f"### {title}")
    for _, row in df.iterrows():
        st.write(f"{row[key_col]}  →  {row[percent_col]}%")
    st.write("")


def render_historical_2d(df_hist):
    st.markdown("### 2D Belakang Historis")
    st.caption(
        "Bagian ini menampilkan 2 digit belakang yang paling sering muncul sebagai lanjutan historis "
        "ketika 2 digit belakang baseline kemarin pernah muncul di data sebelumnya."
    )
    if "Persen" in df_hist.columns:
        for _, row in df_hist.iterrows():
            st.write(f"{row['2D_Belakang']}  →  {row['Persen']}%")
    else:
        for _, row in df_hist.iterrows():
            st.write(f"{row['2D_Belakang']}  →  {row['Frekuensi']}x")
    st.write("")


def main():
    st.title("Mesin Analisis 4D")
    st.caption("Mode utama: baseline memakai hasil 4D kemarin.")

    st.info(
        "Sistem ini ditenagai oleh model komputasi matematis kompleks yang mengekstraksi "
        "matriks transisi dari database historis berskala besar sebagai referensi probabilitas "
        "analitik, bukan sebagai jaminan kepastian mutlak."
    )

    try:
        history = load_data()
    except ValueError as ve:
        st.error(f"Kesalahan data: {ve}")
        return
    except Exception as e:
        st.error(f"Kesalahan tak terduga: {e}")
        return

    st.success(f"{len(history)} entri historis 4D berhasil dimuat.")

    st.subheader("Input Baseline")
    baseline = st.text_input(
        "Masukkan hasil 4D kemarin:",
        max_chars=4,
        placeholder="Contoh: 3506"
    )

    proses = st.button("Proses Analisis", type="primary")

    if not proses:
        return

    baseline = baseline.strip()
    if not baseline.isdigit() or len(baseline) != 4:
        st.error("Input harus tepat 4 digit angka, misalnya 3506.")
        return

    try:
        pairs = build_pairs(history)
        scores = per_position_scores(history, pairs, baseline, recency_halflife=50)
        strong_numbers = generate_strong_numbers(scores)
    except Exception as e:
        st.error(f"Kesalahan perhitungan: {e}")
        return

    target_2d = baseline[2:]
    full_2d = full_2d_markov(pairs, target_2d, top_n=4)

    hasil_4d_4 = to_percent_series(strong_numbers.head(4)[["4D", "Skor_Total"]], "Skor_Total")
    hasil_4d_2 = to_percent_series(strong_numbers.head(2)[["4D", "Skor_Total"]], "Skor_Total")

    hasil_3d_4 = to_percent_series(
        top_unique(strong_numbers.rename(columns={"3D": "Nilai_3D"}), "Nilai_3D", "Skor_Total", 4),
        "Skor_Total"
    )
    hasil_3d_2 = to_percent_series(
        top_unique(strong_numbers.rename(columns={"3D": "Nilai_3D"}), "Nilai_3D", "Skor_Total", 2),
        "Skor_Total"
    )

    hasil_2d_4 = to_percent_series(
        top_unique(strong_numbers, "2D_Belakang", "Skor_Total", 4),
        "Skor_Total"
    )
    hasil_2d_2 = to_percent_series(
        top_unique(strong_numbers, "2D_Belakang", "Skor_Total", 2),
        "Skor_Total"
    )

    if full_2d:
        df_2d_hist = pd.DataFrame(full_2d, columns=["2D_Belakang", "Frekuensi"])
        df_2d_hist["Persen"] = (df_2d_hist["Frekuensi"] / df_2d_hist["Frekuensi"].max() * 100).round(1)
    else:
        df_2d_hist = hasil_2d_4[["2D_Belakang", "Persen"]].copy()

    as_potensial = top_two_digits(scores, 0)
    kop_potensial = top_two_digits(scores, 1)
    kepala_potensial = top_two_digits(scores, 2)
    ekor_potensial = top_two_digits(scores, 3)

    digit_agregat = aggregate_digit_strength(scores)
    bbfs6 = "".join(digit_agregat.head(6)["Digit"].astype(str).tolist())

    st.divider()
    st.header("Skor Atas")

    col1, col2 = st.columns(2)

    with col1:
        render_lines("4D Skor Atas", hasil_4d_4, "4D")
        render_lines("3D Skor Atas", hasil_3d_4, "Nilai_3D")
        render_lines("2D Skor Atas", hasil_2d_4, "2D_Belakang")

    with col2:
        render_historical_2d(df_2d_hist)

    st.divider()
    st.header("Analisa Terbaik")

    render_lines("4D Skor Tertinggi", hasil_4d_2, "4D")
    render_lines("3D Skor Tertinggi", hasil_3d_2, "Nilai_3D")
    render_lines("2D Skor Tertinggi", hasil_2d_2, "2D_Belakang")

    st.divider()
    st.header("Digit Potensial")

    st.write(f"AS POTENSIAL : {as_potensial}")
    st.write(f"KOP POTENSIAL : {kop_potensial}")
    st.write(f"KEPALA POTENSIAL : {kepala_potensial}")
    st.write(f"EKOR POTENSIAL : {ekor_potensial}")
    st.write(f"BBFS : {bbfs6}")


if __name__ == "__main__":
    main()
