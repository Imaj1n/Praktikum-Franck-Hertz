import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from io import BytesIO
from PIL import Image
import pandas as pd

st.set_page_config(layout="wide")
st.title("Ekstraksi Kurva Franck-Hertz dari Gambar")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar kurva Franck-Hertz", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Threshold untuk mengambil piksel terang
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Hilangkan noise kecil (opsional)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Temukan semua kontur putih
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Buat mask kosong untuk menyimpan hanya kurva besar
    mask = np.zeros_like(gray)

    
    pixels = st.number_input("Masukkan sebuah angka", min_value=0, max_value=10000, value=50, step = 1)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > pixels:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Ambil koordinat dari kurva
    y_coords, x_coords = np.where(mask == 255)

    if len(x_coords) == 0:
        st.warning("Kurva tidak ditemukan. Coba unggah gambar lain atau atur parameter.")
    else:
        # Normalisasi koordinat
        x_coords = np.array(x_coords / max(x_coords))
        y_coords = 1 - np.array(y_coords / max(y_coords))

        # Plot kurva hasil deteksi
        fig1, ax1 = plt.subplots()
        ax1.scatter(x_coords, y_coords, s=1, color='red')
        ax1.set_title("Kurva Franck-Hertz (Scatter)")
        ax1.grid(True)

        # Binned statistic
        num_bins = st.slider("Jumlah bin untuk perataan", min_value=50, max_value=1000, value=300, step=50)
        y_binned, bin_edges, _ = binned_statistic(x_coords, y_coords, statistic='mean', bins=num_bins)
        x_binned = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot kurva binned
        fig2, ax2 = plt.subplots()
        ax2.plot(x_binned, y_binned, color='green')
        ax2.set_title("Kurva Franck-Hertz (Binned Mean)")
        ax2.grid(True)

        col1,col2 = st.columns(2)
        col1.pyplot(fig1)
        col2.pyplot(fig2)
        st.markdown("---")
        st.subheader("Skalakan Kurva dengan Data Nyata")

        # Input data nyata dari alat
        colA, colB , colC = st.columns(3)
        dataV_ke_n = colA.number_input("Masukkan Tegangan Nyata (V) pada titik ke-n", value=20.0)
        dataI_ke_n = colB.number_input("Masukkan Arus Nyata (I) pada titik ke-n", value=0.8)
        # Pilih titik ke-n untuk scaling
        n = colC.number_input("Pilih index n (titik referensi)", min_value=0, max_value=len(x_binned)-1, value=2, step=1)

        try:
            scallingx1 = dataV_ke_n / x_binned[n]
            scallingy1 = dataI_ke_n / y_binned[n]

            V_coords = x_binned * scallingx1
            I_coords = y_binned * scallingy1

            # Buat DataFrame
            df_scaled = pd.DataFrame({
                'V_Skala': V_coords,
                'I_Skala': I_coords
            })

            st.dataframe(df_scaled)

            # Unduh CSV
            csv = df_scaled.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Tabel Skala sebagai CSV",
                data=csv,
                file_name='kurva_franck_hertz_terkalibrasi.csv',
                mime='text/csv'
            )
        except ZeroDivisionError:
            st.error("Nilai y_binned[n] atau x_binned[n] = 0, tidak bisa melakukan scaling.")
