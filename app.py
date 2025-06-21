import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("dataset_pddikti_bersih.csv")

# Label encoding
le = LabelEncoder()
df['akreditasi_encoded'] = le.fit_transform(df['akreditasi'])

# Fitur dan normalisasi
features = ["jumlah_semester", "jumlah_mk", "jumlah_sks", "total_sks", "mahasiswa_aktif", "lama_studi_menit"]
X = df[features]
y = df['akreditasi_encoded']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

# PCA for clustering plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Streamlit app layout
st.title("📊 Dashboard Analisis Program Studi PDDIKTI")

st.sidebar.header("Prediksi Akreditasi")
input_data = {}
for f in features:
    input_data[f] = st.sidebar.number_input(f.capitalize().replace("_", " "), float(df[f].min()), float(df[f].max()), float(df[f].mean()))

if st.sidebar.button("Prediksi"):
    user_df = pd.DataFrame([input_data])
    user_scaled = scaler.transform(user_df)
    prediction = model.predict(user_df)
    st.sidebar.success(f"Prediksi Akreditasi: {le.inverse_transform(prediction)[0]}")

# Tabs
tab1, tab2, tab3 = st.tabs(["📄 Data", "📈 Visualisasi", "🎯 Clustering"])

with tab1:
    st.subheader("Data Program Studi")
    st.dataframe(df.head(50))

with tab2:
    st.subheader("Distribusi Akreditasi")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='akreditasi', ax=ax1, palette='Set2')
    st.pyplot(fig1)

    st.subheader("Sebaran Mahasiswa Aktif per Akreditasi")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x='akreditasi', y='mahasiswa_aktif', ax=ax2, palette='Set3')
    ax2.set_yscale("log")
    st.pyplot(fig2)

with tab3:
    st.subheader("Visualisasi Clustering (PCA)")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df["cluster"], palette="Set2", ax=ax3)
    ax3.set_title("PCA Clustering")
    st.pyplot(fig3)
