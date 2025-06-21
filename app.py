import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Page Configuration
st.set_page_config(page_title="Study Program Accreditation Prediction", layout="centered")
st.title("🎓 Study Program Accreditation Prediction & Analysis")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["📊 Dataset & Visualization", "🔍 Clustering", "🧠 Accreditation Prediction"])

# Load Dataset
df = pd.read_csv("dataset_pddikti_bersih.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

features = ['jumlah_semester', 'jumlah_mk', 'jumlah_sks', 'total_sks', 'mahasiswa_aktif']
target = 'akreditasi'

# ---------- Dataset & Visualization Page ----------
if page == "📊 Dataset & Visualization":
    st.subheader("📋 Dataset Table")
    st.dataframe(df, use_container_width=True)

    st.subheader("📌 Accreditation Distribution")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="akreditasi", order=df["akreditasi"].value_counts().index, ax=ax1)
    ax1.set_title("Accreditation Distribution")
    plt.xticks(rotation=0)
    st.pyplot(fig1)

    st.subheader("📈 Feature Correlation")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# ---------- Clustering Page ----------
elif page == "🔍 Clustering":
    st.subheader("🔍 Study Program Clustering (KMeans)")

    df_cluster = df.copy()
    df_cluster_encoded = df_cluster[features].dropna()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_cluster_encoded)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    df_cluster["cluster"] = clusters

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled_data)
    df_cluster["PC1"] = reduced[:, 0]
    df_cluster["PC2"] = reduced[:, 1]

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_cluster, x="PC1", y="PC2", hue="cluster", palette="Set2", ax=ax3)
    ax3.set_title("Study Program Clustering Visualization")
    st.pyplot(fig3)

# ---------- Accreditation Prediction Page ----------
elif page == "🧠 Accreditation Prediction":
    st.subheader("🧠 Study Program Accreditation Prediction")

    if all(col in df.columns for col in features + [target]):
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)

        st.write("📊 Random Forest Model Evaluation")
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax_cm)
        st.pyplot(fig_cm)

        st.markdown("### 🔧 Enter Data for Prediction")
        cols = st.columns(2)
        user_input = {}
        for i, feature in enumerate(features):
            with cols[i % 2]:
                user_input[feature] = st.number_input(
                    f"{feature.replace('_', ' ').title()}",
                    value=float(df[feature].mean())
                )

        if st.button("Predict Accreditation"):
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            pred_label = le.inverse_transform([pred])[0]
            st.success(f"✅ Predicted Accreditation: **{pred_label}**")
    else:
        st.error("🚫 Required columns are missing in the dataset.")
