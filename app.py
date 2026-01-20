import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- KONFIGURASI HALAMAN (HARUS PALING ATAS) ---
st.set_page_config(
    page_title="Simulasi Intensitas Cahaya (Lux) - Bandung",
    layout="wide",
)


# =======================
# 1. FUNGSI DATA & MODEL
# =======================

@st.cache_data
def load_and_prepare_data(csv_path: str = "lux.csv") -> pd.DataFrame:
    """
    Load, praproses, dan siapkan fitur waktu dari file lux.csv.
    Fungsi ini di-cache agar tidak mengulang proses saat app di-refresh.
    """
    df_raw = pd.read_csv(csv_path)

    # Pilih kolom penting dan rename agar lebih jelas
    df = df_raw[["_time", "_value"]].rename(columns={
        "_time": "time",
        "_value": "lux"
    })

    # Konversi ke datetime (UTC) dan buang baris yang invalid (NaT)
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    # Sort berdasarkan waktu
    df = df.sort_values("time").reset_index(drop=True)

    # Interpolasi missing value pada lux (karena ini time series kontinu)
    df["lux"] = df["lux"].interpolate(method="linear")
    df = df.dropna(subset=["lux"]).reset_index(drop=True)

    # Tambah kolom waktu relatif dalam jam
    t0 = df["time"].iloc[0]
    df["time_delta"] = (df["time"] - t0)
    df["time_hours"] = df["time_delta"].dt.total_seconds() / 3600.0

    # Tambah fitur tanggal & jam (untuk EDA pola harian)
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour + df["time"].dt.minute / 60.0

    return df


def L_eq(t_hours: np.ndarray, A: float, C: float, phi: float) -> np.ndarray:
    """
    Fungsi kesetimbangan (equilibrium) harian:
        L_eq(t) = max(0, A * sin(2π t / 24 + φ) + C)
    """
    t_hours = np.asarray(t_hours)
    omega = 2 * np.pi / 24.0  # periode 24 jam
    raw = A * np.sin(omega * t_hours + phi) + C
    return np.maximum(0.0, raw)


def dL_dt(t: float, L: float, k: float, A: float, C: float, phi: float) -> float:
    """
    ODE: dL/dt = -k (L - L_eq(t))
    """
    return -k * (L - float(L_eq(t, A, C, phi)))


def rk4_step(t: float, L: float, dt: float,
             k: float, A: float, C: float, phi: float) -> float:
    """
    Satu langkah metode Runge–Kutta orde 4 (RK4) untuk ODE skalar L(t).
    """
    k1 = dL_dt(t, L, k, A, C, phi)
    k2 = dL_dt(t + 0.5 * dt, L + 0.5 * dt * k1, k, A, C, phi)
    k3 = dL_dt(t + 0.5 * dt, L + 0.5 * dt * k2, k, A, C, phi)
    k4 = dL_dt(t + dt, L + dt * k3, k, A, C, phi)
    return L + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_lux(time_hours: np.ndarray,
                 L0: float,
                 k: float, A: float, C: float, phi: float) -> np.ndarray:
    """
    Mensimulasikan L(t) pada grid waktu time_hours menggunakan RK4.
    """
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
    Melakukan grid search sederhana untuk mencari (k, A, C, phi) terbaik
    berdasarkan MSE pada data training.

    Di-cache supaya hanya dihitung sekali saat app dijalankan.
    """
    lux_max = lux_data.max()

    # Rentang parameter (bisa kamu jelaskan di laporan)
    k_values = np.linspace(0.05, 0.8, 6)                         # 6 nilai
    A_values = np.linspace(0.5 * lux_max, 1.5 * lux_max, 6)      # 6 nilai
    C_values = np.linspace(-0.1 * lux_max, 0.3 * lux_max, 5)     # 5 nilai
    phi_values = np.linspace(-np.pi, np.pi, 7)                    # 7 nilai

    best_params = None
    best_loss = np.inf

    L0 = lux_data[0]

    # Loop grid (1260 kombinasi)
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


def run_simulation_df(df: pd.DataFrame,
                      k: float, A: float, C: float, phi: float) -> pd.DataFrame:
    """
    Membungkus simulasi ke dalam DataFrame:
    - time
    - lux_data
    - lux_sim
    - L_eq (garis kesetimbangan) → untuk visualisasi tambahan
    """
    time_hours = df["time_hours"].values
    lux_data = df["lux"].values
    L0 = lux_data[0]

    lux_sim = simulate_lux(time_hours, L0, k, A, C, phi)
    l_eq = L_eq(time_hours, A, C, phi)

    result = pd.DataFrame({
        "time": df["time"].values,
        "time_hours": time_hours,
        "lux_data": lux_data,
        "lux_sim": lux_sim,
        "lux_eq": l_eq,
    })
    return result


# =======================
# 2. MAIN APP STREAMLIT
# =======================

def main():
    # ---------- SIDEBAR ----------
    st.sidebar.title("Pengaturan Simulasi")

    st.sidebar.markdown(
        """
        Aplikasi ini memodelkan **intensitas cahaya (lux)** di Bandung
        menggunakan:

        - **Model ODE kontinu**
        - **Metode numerik RK4**
        - Data sensor IoT nyata (`lux.csv`)
        """
    )

    # Load data
    df = load_and_prepare_data("lux.csv")
    time_hours = df["time_hours"].values
    lux_data = df["lux"].values

    # Info data di sidebar
    st.sidebar.subheader("Ringkasan Data")
    st.sidebar.write(f"Jumlah observasi: **{len(df)}**")
    st.sidebar.write(
        f"Rentang waktu: **{df['time'].iloc[0]}** s.d. **{df['time'].iloc[-1]}**"
    )
    st.sidebar.write(
        f"Lux min/max: **{lux_data.min():.1f} / {lux_data.max():.1f}**"
    )

    # Pilihan mode parameter
    mode = st.sidebar.radio(
        "Mode Parameter Model",
        ("Best Fit (Otomatis)", "Manual (Slider)"),
    )

    # Bagian train/test split (fix 2 hari pertama untuk train)
    max_train_hours = 48.0
    train_mask = time_hours <= max_train_hours
    test_mask = time_hours > max_train_hours

    # ---------- FIT OTOMATIS ----------
    if mode == "Best Fit (Otomatis)":
        st.sidebar.subheader("Pengaturan Training")
        st.sidebar.write(
            "• Data training: jam 0 s.d. 48\n"
            "• Data testing: sisanya"
        )

        with st.spinner("Menghitung parameter terbaik (grid search)..."):
            best_params, best_loss = grid_search_params(
                time_hours, lux_data, train_mask
            )

        k_best, A_best, C_best, phi_best = best_params

        st.sidebar.markdown("**Parameter terbaik (hasil fitting):**")
        st.sidebar.write(f"- k   = `{k_best:.4f}`")
        st.sidebar.write(f"- A   = `{A_best:.1f}`")
        st.sidebar.write(f"- C   = `{C_best:.1f}`")
        st.sidebar.write(f"- phi = `{phi_best:.3f} rad`")

        # Pakai parameter terbaik untuk simulasi
        k, A, C, phi = k_best, A_best, C_best, phi_best

    # ---------- MODE MANUAL ----------
    else:
        st.sidebar.subheader("Pengaturan Slider (Manual)")

        lux_max = float(lux_data.max())

        k = st.sidebar.slider("k (kecepatan relaksasi)", 0.01, 1.0, 0.3, 0.01)
        A = st.sidebar.slider("A (amplitudo)", 0.0, 2.0 * lux_max, lux_max, lux_max / 20)
        C = st.sidebar.slider(
            "C (offset)",
            -0.5 * lux_max,
            0.5 * lux_max,
            0.0,
            lux_max / 20,
        )
        phi = st.sidebar.slider(
            "phi (fase, radian)",
            -3.14,
            3.14,
            0.0,
            0.1,
        )

        st.sidebar.markdown(
            "_Catatan: gunakan slider ini untuk demo **what-if** saat presentasi._"
        )

    # Jalankan simulasi dengan parameter (baik best-fit atau manual)
    df_sim = run_simulation_df(df, k, A, C, phi)
    residuals = df_sim["lux_data"] - df_sim["lux_sim"]

    # Hitung metrik error
    mse_total = mse(df_sim["lux_data"], df_sim["lux_sim"])
    rmse_total = rmse(df_sim["lux_data"], df_sim["lux_sim"])
    r2_total = r2_score(df_sim["lux_data"], df_sim["lux_sim"])

    mse_train = mse(df_sim["lux_data"][train_mask], df_sim["lux_sim"][train_mask])
    mse_test = mse(df_sim["lux_data"][test_mask], df_sim["lux_sim"][test_mask])

    # ---------- KONTEN UTAMA ----------
    st.title("Simulasi Dinamika Intensitas Cahaya (Lux) - Bandung")

    st.markdown(
        """
        Aplikasi ini merupakan bagian dari **Tugas Besar Pemodelan (Jalur 1)**:

        - Sistem dinamis **kontinu** berbasis **persamaan diferensial (ODE)**
        - Diselesaikan dengan **metode Runge–Kutta Orde 4 (RK4)**
        - Divalidasi dengan **data time series riil** intensitas cahaya (lux)
        """
    )

    # Tampilkan metrik utama (sebagai highlight)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MSE (total)", f"{mse_total:.2f}")
    col2.metric("RMSE (total)", f"{rmse_total:.2f}")
    col3.metric("R² (total)", f"{r2_total:.3f}")
    col4.metric("MSE Train / Test", f"{mse_train:.1f} / {mse_test:.1f}")

    # Tabs: Simulasi | EDA | Residual & Download
    tab_sim, tab_eda, tab_res = st.tabs(
        ["🔮 Simulasi Model", "📊 EDA Data Lux", "📉 Residual & Download"]
    )

    # ---------- TAB SIMULASI ----------
    with tab_sim:
        st.subheader("Perbandingan Data Riil vs Simulasi Model")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_sim["time"], df_sim["lux_data"], label="Data riil (lux)")
        ax.plot(df_sim["time"], df_sim["lux_sim"], label="Simulasi ODE (RK4)", linestyle="--")
        ax.set_xlabel("Waktu")
        ax.set_ylabel("Lux")
        ax.set_title("Data Riil vs Simulasi")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Garis Kesetimbangan Harian L_eq(t)")

        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(df_sim["time"], df_sim["lux_eq"], label="L_eq(t) (equilibrium)", alpha=0.7)
        ax2.set_xlabel("Waktu")
        ax2.set_ylabel("Lux")
        ax2.set_title("Fungsi Kesetimbangan Harian")
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2)

        st.markdown(
            """
            **Interpretasi singkat:**

            - Kurva simulasi (putus-putus) menunjukkan solusi ODE kita.
            - Jika kurva simulasi cukup mengikuti pola data riil, model dianggap cukup representatif.
            - Garis **L_eq(t)** menggambarkan pola siang–malam ideal yang ingin dicapai sistem.
            """
        )

    # ---------- TAB EDA ----------
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

        st.markdown(
            """
            Dari pola harian di atas terlihat:

            - Lux hampir 0 pada malam hari.
            - Naik tajam saat pagi–siang.
            - Turun kembali menjelang sore–malam.

            Pola ini yang kita tangkap dengan fungsi **L_eq(t)** di model ODE.
            """
        )

    # ---------- TAB RESIDUAL & DOWNLOAD ----------
    with tab_res:
        st.subheader("Residual (Data - Simulasi)")

        fig5, ax5 = plt.subplots(figsize=(10, 3))
        ax5.plot(df_sim["time"], residuals)
        ax5.axhline(0.0, linestyle="--")
        ax5.set_xlabel("Waktu")
        ax5.set_ylabel("Residual (lux)")
        ax5.set_title("Residual Model terhadap Data")
        fig5.tight_layout()
        st.pyplot(fig5)

        st.markdown(
            """
            Residual yang **acak di sekitar 0** menandakan model cukup baik.
            Jika ada pola tertentu (misalnya selalu under/over di jam tertentu),
            itu bisa dijadikan bahan **pembahasan kritis** di laporan.
            """
        )

        st.markdown("---")
        st.subheader("Download Hasil Simulasi")

        csv_buffer = df_sim.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV Simulasi (time, lux_data, lux_sim, lux_eq)",
            data=csv_buffer,
            file_name="simulasi_lux_model_ode.csv",
            mime="text/csv",
        )

        st.markdown(
            """
            File ini bisa kamu gunakan untuk:

            - Lampiran tambahan di laporan
            - Analisis lanjutan (misalnya di Excel / notebook lain)
            - Bukti bahwa model benar-benar menghasilkan output kuantitatif
            """
        )


if __name__ == "__main__":
    main()
