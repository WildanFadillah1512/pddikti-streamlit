import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Judul
st.title("🎓 Prediksi Akreditasi Program Studi")

# Load dan bersihkan data
df = pd.read_csv("dataset_pddikti_bersih.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Debug: tampilkan kolom yang tersedia
st.subheader("📋 Kolom tersedia dalam dataset:")
st.write(df.columns.tolist())

# Pilih fitur dan target yang valid
features = ['jumlah_semester', 'jumlah_mk', 'jumlah_sks', 'total_sks', 'mahasiswa_aktif']
target = 'akreditasi'

# Pastikan semua kolom tersedia
if all(col in df.columns for col in features + [target]):
    # Encode target
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])

    # Ambil data
    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standarisasi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluasi
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.subheader("📊 Evaluasi Model")
    st.write(pd.DataFrame(report).transpose())

    # Visualisasi Confusion Matrix
    st.subheader("🧩 Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax)
    st.pyplot(fig)

    # Form input user untuk prediksi
    st.subheader("📝 Coba Prediksi Sendiri")

    user_input = {}
    for feature in features:
        val = st.number_input(f"Masukkan nilai untuk {feature}", value=float(df[feature].mean()))
        user_input[feature] = val

    if st.button("Prediksi Akreditasi"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        pred_label = le.inverse_transform([pred])[0]
        st.success(f"✅ Prediksi Akreditasi: **{pred_label}**")

else:
    st.error("🚫 Kolom yang dibutuhkan tidak tersedia dalam dataset. Periksa nama kolom atau dataset.")
