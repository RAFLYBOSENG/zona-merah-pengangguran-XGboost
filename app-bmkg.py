import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- CONFIG STREAMLIT ---
st.set_page_config(
    page_title="Simulasi Lux + BMKG - Bandung",
    layout="wide",
)


# ============================
# 1. FUNGSI DATA & MODEL LUX
# ============================

@st.cache_data
def load_lux_data() -> pd.DataFrame:
    """
    Load, praproses, dan siapkan fitur waktu dari file dataset/lux.csv
    (relatif terhadap lokasi app.py).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "dataset", "lux.csv")

    df_raw = pd.read_csv(csv_path)

    # Pilih kolom penting dan rename
    df = df_raw[["_time", "_value"]].rename(columns={
        "_time": "time",
        "_value": "lux"
    })

    # Konversi waktu ke datetime UTC
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    # Sort
    df = df.sort_values("time").reset_index(drop=True)

    # Interpolasi lux dan drop NaN
    df["lux"] = df["lux"].interpolate(method="linear")
    df = df.dropna(subset=["lux"]).reset_index(drop=True)

    # Tambah time_hours relatif
    t0 = df["time"].iloc[0]
    df["time_delta"] = (df["time"] - t0)
    df["time_hours"] = df["time_delta"].dt.total_seconds() / 3600.0

    # Fitur tanggal & jam
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60.0

    return df


def L_eq(t_hours: np.ndarray, A: float, C: float, phi: float) -> np.ndarray:
    """
    Fungsi kesetimbangan (equilibrium) harian:
    L_eq(t) = max(0, A * sin(2π t / 24 + φ) + C)
    """
    t_hours = np.asarray(t_hours)
    omega = 2 * np.pi / 24.0
    raw = A * np.sin(omega * t_hours + phi) + C
    return np.maximum(0.0, raw)


def dL_dt(t: float, L: float, k: float, A: float, C: float, phi: float) -> float:
    """ODE: dL/dt = -k (L - L_eq(t))"""
    return -k * (L - float(L_eq(t, A, C, phi)))


def rk4_step(t: float, L: float, dt: float,
             k: float, A: float, C: float, phi: float) -> float:
    """Satu langkah RK4 untuk ODE skalar."""
    k1 = dL_dt(t, L, k, A, C, phi)
    k2 = dL_dt(t + 0.5 * dt, L + 0.5 * dt * k1, k, A, C, phi)
    k3 = dL_dt(t + 0.5 * dt, L + 0.5 * dt * k2, k, A, C, phi)
    k4 = dL_dt(t + dt, L + dt * k3, k, A, C, phi)
    return L + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_lux(time_hours: np.ndarray,
                 L0: float,
                 k: float, A: float, C: float, phi: float) -> np.ndarray:
    """Simulasikan L(t) di grid time_hours memakai RK4."""
    time_hours = np.asarray(time_hours)
    n = len(time_hours)
    L_sim = np.zeros(n)
    L_sim[0] = L0

    for i in range(n - 1):
        t = time_hours[i]
        dt = time_hours[i + 1] - time_hours[i]
        L_sim[i + 1] = rk4_step(t, L_sim[i], dt, k, A, C, phi)

    return L_sim


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


@st.cache_data
def grid_search_params(time_hours: np.ndarray,
                       lux_data: np.ndarray,
                       train_mask: np.ndarray):
    """
    Grid search sederhana untuk (k, A, C, phi) berdasarkan MSE di data train.
    Di-cache supaya tidak diulang-ulang.
    """
    lux_max = lux_data.max()

    k_values = np.linspace(0.05, 0.8, 6)
    A_values = np.linspace(0.5 * lux_max, 1.5 * lux_max, 6)
    C_values = np.linspace(-0.1 * lux_max, 0.3 * lux_max, 5)
    phi_values = np.linspace(-np.pi, np.pi, 7)

    best_params = None
    best_loss = np.inf

    L0 = lux_data[0]

    for k in k_values:
        for A in A_values:
            for C in C_values:
                for phi in phi_values:
                    L_sim = simulate_lux(time_hours, L0, k, A, C, phi)
                    loss = mse(lux_data[train_mask], L_sim[train_mask])
                    if loss < best_loss:
                        best_loss = loss
                        best_params = (k, A, C, phi)

    return best_params, best_loss


# ============================
# 2. FUNGSI BMKG API
# ============================

@st.cache_data
def fetch_bmkg_forecast(adm4: str = "32.04.29.2001") -> pd.DataFrame | None:
    """
    Mengambil prakiraan cuaca BMKG untuk kode adm4 tertentu
    dan mengubahnya menjadi DataFrame yang relatif flat.
    """
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm4}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.warning(f"Gagal mengambil data BMKG: {e}")
        return None

    lokasi_global = data.get("lokasi", {})
    records: list[dict] = []

    for entry in data.get("data", []):
        lokasi_entry = entry.get("lokasi", {})
        cuaca_lists = entry.get("cuaca", [])
        for daily_list in cuaca_lists:
            for row in daily_list:
                rec = {}
                rec.update(lokasi_global)
                rec.update(lokasi_entry)
                rec.update(row)
                records.append(rec)

    if not records:
        return None

    df_bmkg = pd.DataFrame(records)

    # Konversi kolom datetime yg ada
    for col in ["datetime", "utc_datetime", "local_datetime", "analysis_date"]:
        if col in df_bmkg.columns:
            df_bmkg[col] = pd.to_datetime(df_bmkg[col], errors="coerce")

    if "local_datetime" in df_bmkg.columns:
        df_bmkg = df_bmkg.sort_values("local_datetime").reset_index(drop=True)

    return df_bmkg


# ============================
# 3. MAIN STREAMLIT APP
# ============================

def main():
    st.title("Simulasi Dinamika Lux + BMKG - Kabupaten Bandung")

    st.markdown(
        """
        Aplikasi ini merupakan bagian dari **Tugas Besar Pemodelan (Jalur 1)**:

        - Model **ODE kontinu** untuk intensitas cahaya (lux)
        - Penyelesaian numerik dengan **Runge–Kutta Orde 4 (RK4)**
        - Data utama: **sensor IoT (lux.csv)**
        - Data tambahan: **prakiraan cuaca BMKG (Kab. Bandung, adm4=32.04.29.2001)**
        """
    )

    # --- LOAD DATA LUX ---
    df = load_lux_data()
    time_hours = df["time_hours"].values
    lux_data = df["lux"].values

    # Train/test split (2 hari pertama untuk train)
    max_train_hours = 48.0
    train_mask = time_hours <= max_train_hours
    test_mask = time_hours > max_train_hours

    # --- SIDEBAR ---
    st.sidebar.header("Pengaturan")

    st.sidebar.subheader("Ringkasan Data Lux")
    st.sidebar.write(f"Jumlah observasi: **{len(df)}**")
    st.sidebar.write(
        f"Rentang waktu: **{df['time'].iloc[0]}** s.d. **{df['time'].iloc[-1]}**"
    )
    st.sidebar.write(
        f"Lux min/max: **{lux_data.min():.1f} / {lux_data.max():.1f}**"
    )

    mode = st.sidebar.radio(
        "Mode Parameter Model",
        ("Best Fit (Otomatis)", "Manual (Slider)"),
    )

    st.sidebar.subheader("Prediksi ke Depan")
    horizon_plot_hours = st.sidebar.slider(
        "Horizon prediksi (untuk grafik) [jam]",
        min_value=0.0,
        max_value=48.0,
        value=0.0,
        step=1.0,
        help="Berapa jam ke depan dari data terakhir yang ingin ditampilkan di grafik.",
    )

    user_pred_hours = st.sidebar.number_input(
        "Jam ke depan untuk prediksi lux (nilai tunggal)",
        min_value=0.0,
        max_value=48.0,
        value=0.0,
        step=1.0,
        help="Isi > 0 untuk melihat nilai prediksi lux pada jam tertentu setelah data terakhir.",
    )

    # --- PARAMETER (BEST FIT / MANUAL) ---
    if mode == "Best Fit (Otomatis)":
        st.sidebar.subheader("Training")
        st.sidebar.write("Data training: jam 0–48, sisanya testing.")

        with st.spinner("Melakukan grid search parameter terbaik..."):
            best_params, best_loss = grid_search_params(
                time_hours, lux_data, train_mask
            )

        k_best, A_best, C_best, phi_best = best_params

        st.sidebar.markdown("**Parameter terbaik (hasil fitting):**")
        st.sidebar.write(f"- k   = `{k_best:.4f}`")
        st.sidebar.write(f"- A   = `{A_best:.1f}`")
        st.sidebar.write(f"- C   = `{C_best:.1f}`")
        st.sidebar.write(f"- phi = `{phi_best:.3f} rad`")

        k, A, C, phi = k_best, A_best, C_best, phi_best

    else:
        st.sidebar.subheader("Pengaturan Slider (Manual)")

        lux_max = float(lux_data.max())
        lux_step = lux_max / 20 if lux_max > 0 else 1.0

        k = st.sidebar.slider(
            "k (kecepatan relaksasi)", 0.01, 1.0, 0.3, 0.01
        )
        A = st.sidebar.slider(
            "A (amplitudo)",
            0.0,
            2.0 * lux_max,
            lux_max,
            lux_step,
        )
        C = st.sidebar.slider(
            "C (offset)",
            -0.5 * lux_max,
            0.5 * lux_max,
            0.0,
            lux_step,
        )
        phi = st.sidebar.slider(
            "phi (fase, radian)",
            -3.14,
            3.14,
            0.0,
            0.1,
        )

    # --- GRID WAKTU DIPERPANJANG UNTUK PREDIKSI ---
    dt_median = float(np.median(np.diff(time_hours)))
    horizon_total = max(horizon_plot_hours, user_pred_hours, 0.0)

    if horizon_total > 0:
        n_future = int(np.ceil(horizon_total / dt_median))
        t_last = time_hours[-1]
        t_future = t_last + dt_median * np.arange(1, n_future + 1)
        time_hours_ext = np.concatenate([time_hours, t_future])
    else:
        time_hours_ext = time_hours

    # Simulasi di grid extended
    L0 = lux_data[0]
    lux_sim_ext = simulate_lux(time_hours_ext, L0, k, A, C, phi)
    lux_eq_ext = L_eq(time_hours_ext, A, C, phi)

    # Bangun df_sim extended
    time0 = df["time"].iloc[0]
    time_ext = pd.to_timedelta(time_hours_ext, unit="h") + time0

    lux_data_ext = np.full_like(time_hours_ext, np.nan, dtype=float)
    lux_data_ext[: len(lux_data)] = lux_data

    df_sim = pd.DataFrame({
        "time": time_ext,
        "time_hours": time_hours_ext,
        "lux_data": lux_data_ext,
        "lux_sim": lux_sim_ext,
        "lux_eq": lux_eq_ext,
    })

    # Mask bagian yang punya data riil
    mask_have_data = ~np.isnan(df_sim["lux_data"].values)

    mse_total = mse(df_sim["lux_data"][mask_have_data], df_sim["lux_sim"][mask_have_data])
    rmse_total = rmse(df_sim["lux_data"][mask_have_data], df_sim["lux_sim"][mask_have_data])
    r2_total = r2_score(df_sim["lux_data"][mask_have_data], df_sim["lux_sim"][mask_have_data])

    # Train/test metric pakai indeks asli
    mse_train = mse(lux_data[train_mask], lux_sim_ext[train_mask])
    mse_test = mse(lux_data[test_mask], lux_sim_ext[test_mask])

    # --- METRIK DI ATAS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MSE (total)", f"{mse_total:.2f}")
    col2.metric("RMSE (total)", f"{rmse_total:.2f}")
    col3.metric("R² (total)", f"{r2_total:.3f}")
    col4.metric("MSE Train / Test", f"{mse_train:.1f} / {mse_test:.1f}")

    # --- PREDIKSI TUNGGAL DARI INPUT USER ---
    if user_pred_hours > 0:
        target_time_hours = time_hours[-1] + user_pred_hours
        # cari index waktu terdekat
        idx_pred = int(np.argmin(np.abs(time_hours_ext - target_time_hours)))
        pred_time = df_sim["time"].iloc[idx_pred]
        pred_lux = df_sim["lux_sim"].iloc[idx_pred]

        st.info(
            f"Prediksi lux **{user_pred_hours:.0f} jam** setelah data terakhir "
            f"(sekitar **{pred_time}**) adalah **{pred_lux:.2f} lux**."
        )

    # --- FETCH BMKG ---
    df_bmkg = fetch_bmkg_forecast("32.04.29.2001")

    # --- TAB: SIMULASI | EDA LUX | BMKG | RESIDUAL ---
    tab_sim, tab_eda, tab_bmkg, tab_res = st.tabs(
        ["🔮 Simulasi & Prediksi", "📊 EDA Data Lux", "🌦️ Data BMKG", "📉 Residual & Download"]
    )

    # ==================================
    # TAB 1: SIMULASI & PREDIKSI
    # ==================================
    with tab_sim:
        st.subheader("Perbandingan Data Riil vs Simulasi (termasuk prediksi)")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_sim["time"], df_sim["lux_sim"], label="Simulasi ODE (RK4)", linestyle="--")

        # Plot bagian data riil (yang bukan NaN)
        ax.plot(df["time"], df["lux"], label="Data riil (lux)")

        # Garis batas antara data historis dan prediksi
        t_last_data = df["time"].iloc[-1]
        ax.axvline(t_last_data, linestyle=":", label="Batas data historis")

        ax.set_xlabel("Waktu")
        ax.set_ylabel("Lux")
        ax.set_title("Data Riil vs Simulasi Model ODE")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown(
            """
            **Catatan:**

            - Sebelum garis putus-putus vertikal: model di-*fit* ke data historis.
            - Setelah garis tersebut: model diekstrapolasi → **prediksi lux** untuk jam-jam berikutnya.
            - Parameter model berasal dari mode yang dipilih (Best Fit atau Manual).
            """
        )

        st.markdown("---")
        st.subheader("Garis Kesetimbangan Harian L_eq(t)")

        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(df_sim["time"], df_sim["lux_eq"], label="L_eq(t)")
        ax2.set_xlabel("Waktu")
        ax2.set_ylabel("Lux")
        ax2.set_title("Fungsi Kesetimbangan Harian")
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2)

    # ==================================
    # TAB 2: EDA LUX
    # ==================================
    with tab_eda:
        st.subheader("Statistik Deskriptif Data Lux")
        st.write(df[["lux"]].describe())

        st.markdown("---")
        st.subheader("Time Series Lux")

        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.plot(df["time"], df["lux"])
        ax3.set_xlabel("Waktu")
        ax3.set_ylabel("Lux")
        ax3.set_title("Time Series Intensitas Cahaya (Lux)")
        fig3.tight_layout()
        st.pyplot(fig3)

        st.markdown("---")
        st.subheader("Pola Harian per Tanggal")

        selected_dates = sorted(df["date"].unique())
        date_to_highlight = st.selectbox(
            "Pilih tanggal untuk di-highlight:",
            selected_dates,
            index=0,
        )

        fig4, ax4 = plt.subplots(figsize=(10, 3))
        for d, sub in df.groupby("date"):
            label = str(d)
            alpha = 1.0 if d == date_to_highlight else 0.3
            lw = 2.0 if d == date_to_highlight else 1.0
            ax4.plot(sub["hour"], sub["lux"], label=label, alpha=alpha, linewidth=lw)

        ax4.set_xlabel("Jam (lokal)")
        ax4.set_ylabel("Lux")
        ax4.set_title("Pola Harian Lux per Tanggal")
        ax4.legend(title="Tanggal", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig4.tight_layout()
        st.pyplot(fig4)

    # ==================================
    # TAB 3: BMKG
    # ==================================
    with tab_bmkg:
        st.subheader("Prakiraan Cuaca BMKG - Kab. Bandung (adm4=32.04.29.2001)")

        if df_bmkg is None or df_bmkg.empty:
            st.warning("Data BMKG tidak tersedia atau gagal diambil.")
        else:
            # Tampilkan beberapa kolom penting
            cols_show = [c for c in ["local_datetime", "t", "hu", "tcc", "tp", "weather_desc"] if c in df_bmkg.columns]
            st.write(df_bmkg[cols_show].head())

            st.markdown("---")
            st.subheader("Time Series Suhu, Tutupan Awan, dan Curah Hujan (prakiraan)")

            fig_b, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            if "t" in df_bmkg.columns:
                axs[0].plot(df_bmkg["local_datetime"], df_bmkg["t"])
                axs[0].set_ylabel("Suhu (°C)")
                axs[0].set_title("Prakiraan Suhu Udara (BMKG)")

            if "tcc" in df_bmkg.columns:
                axs[1].plot(df_bmkg["local_datetime"], df_bmkg["tcc"])
                axs[1].set_ylabel("Tutupan Awan (%)")
                axs[1].set_title("Prakiraan Tutupan Awan (BMKG)")

            if "tp" in df_bmkg.columns:
                axs[2].plot(df_bmkg["local_datetime"], df_bmkg["tp"])
                axs[2].set_ylabel("Curah Hujan (mm)")
                axs[2].set_title("Prakiraan Curah Hujan (BMKG)")

            axs[2].set_xlabel("Waktu Lokal")
            fig_b.tight_layout()
            st.pyplot(fig_b)

            st.markdown(
                """
                **Ide pembahasan di laporan:**

                - Bandingkan pola terang–gelap (lux) dengan prakiraan tutupan awan (`tcc`) dan hujan (`tp`).
                - Jelaskan bahwa model lux yang digunakan di sini belum secara eksplisit memasukkan `tcc`/`tp`,
                  sehingga pengaruh awan & hujan muncul sebagai error/residual.
                """
            )

    # ==================================
    # TAB 4: RESIDUAL & DOWNLOAD
    # ==================================
    with tab_res:
        st.subheader("Residual (Data - Simulasi)")

        residuals = df_sim["lux_data"].copy()
        residuals[mask_have_data] = df_sim["lux_data"][mask_have_data] - df_sim["lux_sim"][mask_have_data]

        fig5, ax5 = plt.subplots(figsize=(10, 3))
        ax5.plot(df_sim["time"][mask_have_data], residuals[mask_have_data])
        ax5.axhline(0.0, linestyle="--")
        ax5.set_xlabel("Waktu")
        ax5.set_ylabel("Residual (lux)")
        ax5.set_title("Residual Model terhadap Data")
        fig5.tight_layout()
        st.pyplot(fig5)

        st.markdown(
            """
            Residual yang acak di sekitar 0 → model cukup baik.  
            Jika ada pola tertentu, bisa dijadikan bahan kritik model
            (misalnya efek awan/hujan yang belum dimodelkan).
            """
        )

        st.markdown("---")
        st.subheader("Download Hasil Simulasi (termasuk prediksi)")

        csv_buffer = df_sim.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV Simulasi (time, lux_data, lux_sim, lux_eq)",
            data=csv_buffer,
            file_name="simulasi_lux_model_ode_ext.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
