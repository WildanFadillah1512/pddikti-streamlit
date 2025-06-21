import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# -----------------------------
# Load dan Preprocess Dataset
# -----------------------------
df = pd.read_csv("dataset_pddikti_bersih.csv")

features = ["jumlah_semester", "jumlah_mk", "jumlah_sks", "total_sks", "mahasiswa_aktif", "lama_studi"]

le = LabelEncoder()
df["akreditasi_encoded"] = le.fit_transform(df["akreditasi"])

X = df[features]
y = df["akreditasi_encoded"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# -----------------------------
# UI Streamlit
# -----------------------------
st.set_page_config(page_title="Prediksi Akreditasi Prodi", layout="wide")
st.title("🎓 Prediksi Akreditasi Program Studi PDDIKTI")
st.markdown("Dashboard interaktif untuk analisis dan prediksi akreditasi program studi menggunakan machine learning.")

# -----------------------------
# Visualisasi Distribusi Akreditasi
# -----------------------------
st.subheader("📊 Visualisasi Distribusi Akreditasi")
fig, ax = plt.subplots()
sns.countplot(x="akreditasi", data=df, order=sorted(df["akreditasi"].unique()), ax=ax)
st.pyplot(fig)

# -----------------------------
# Visualisasi Korelasi Fitur
# -----------------------------
st.subheader("📈 Korelasi Antar Fitur")
fig2, ax2 = plt.subplots()
sns.heatmap(df[features].corr(), annot=True, cmap="YlGnBu", ax=ax2)
st.pyplot(fig2)

# -----------------------------
# Fitur Input User
# -----------------------------
st.sidebar.header("🧠 Input Data untuk Prediksi Akreditasi")
user_input = {}
for col in features:
    user_input[col] = st.sidebar.number_input(
        f"{col.replace('_', ' ').capitalize()}",
        min_value=float(df[col].min()),
        max_value=float(df[col].max()),
        value=float(df[col].mean())
    )

if st.sidebar.button("Prediksi"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)
    akreditasi_pred = le.inverse_transform(pred)[0]
    st.sidebar.success(f"Hasil Prediksi Akreditasi: **{akreditasi_pred}**")
