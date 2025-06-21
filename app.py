import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Judul
st.title("🎓 Prediksi & Analisis Program Studi PDDIKTI")

# Load dan bersihkan data
df = pd.read_csv("dataset_pddikti_bersih.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Sidebar
st.sidebar.header("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["📊 Dataset & Visualisasi", "🔍 Clustering", "🧠 Prediksi Akreditasi"])

# Fitur dan Target
features = ['jumlah_semester', 'jumlah_mk', 'jumlah_sks', 'total_sks', 'mahasiswa_aktif']
target = 'akreditasi'

# Cek validitas kolom
if not all(col in df.columns for col in features + [target]):
    st.error("❌ Kolom penting tidak ditemukan dalam dataset.")
    st.stop()

# Encode target
le = LabelEncoder()
df[target] = le.fit_transform(df[target])

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ==========================
# Halaman 1: Dataset & Visualisasi
# ==========================
if page == "📊 Dataset & Visualisasi":
    st.subheader("📋 Dataset PDDIKTI")
    st.dataframe(df)

    st.subheader("📌 Distribusi Akreditasi")
    fig1, ax1 = plt.subplots()
    df['akreditasi'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_xlabel("Akreditasi (Encoded)")
    ax1.set_ylabel("Jumlah")
    st.pyplot(fig1)

    st.subheader("📈 Korelasi Antar Fitur")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df[features + [target]].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

# ==========================
# Halaman 2: Clustering
# ==========================
elif page == "🔍 Clustering":
    st.subheader("🔎 Clustering KMeans")
    k = st.slider("Pilih jumlah klaster", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # PCA untuk reduksi dimensi ke 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['cluster'] = clusters

    fig3, ax3 = plt.subplots()
    scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    ax3.set_title("Visualisasi Clustering (PCA)")
    st.pyplot(fig3)

    st.write("Jumlah data per klaster:")
    st.write(df['cluster'].value_counts())

# ==========================
# Halaman 3: Prediksi Akreditasi
# ==========================
elif page == "🧠 Prediksi Akreditasi":
    st.subheader("📌 Prediksi Akreditasi Program Studi")

    # Split data
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.write("📊 Evaluasi Model:")
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("🧩 Confusion Matrix")
    fig4, ax4 = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax4)
    st.pyplot(fig4)

    # Input user
    st.subheader("📝 Coba Prediksi Sendiri")

    user_input = {}
    for feature in features:
        val = st.number_input(f"Masukkan nilai untuk {feature}", value=float(df[feature].mean()))
        user_input[feature] = val

    if st.button("Prediksi Akreditasi"):
        input_df = pd.DataFrame([user_input])
        pred = model.predict(input_df)[0]
        pred_label = le.inverse_transform([pred])[0]
        st.success(f"✅ Prediksi Akreditasi: **{pred_label}**")
