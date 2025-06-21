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

# Page configuration
st.set_page_config(page_title="PDDIKTI Prediction", layout="wide")
st.title("🎓 Prediction & Analysis of PDDIKTI Study Programs")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["📊 Dataset & Visualization", "🔍 Clustering", "🧠 Accreditation Prediction"])

# Load and clean data
df = pd.read_csv("dataset_pddikti_bersih.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Features and target
features = ['jumlah_semester', 'jumlah_mk', 'jumlah_sks', 'total_sks', 'mahasiswa_aktif']
target = 'akreditasi'

# ---------------------- Page 1 -------------------------
if page == "📊 Dataset & Visualization":
    st.subheader("📋 PDDIKTI Dataset")
    st.dataframe(df)

    st.subheader("📌 Accreditation Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="akreditasi", order=df["akreditasi"].value_counts().index, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("📈 Feature Correlation")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# ---------------------- Page 2 -------------------------
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

    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df_cluster, x="PC1", y="PC2", hue="cluster", palette="Set2", ax=ax3)
    st.pyplot(fig3)

# ---------------------- Page 3 -------------------------
elif page == "🧠 Accreditation Prediction":
    st.subheader("📝 Try Predicting Study Program Accreditation")

    if all(col in df.columns for col in features + [target]):
        # Encode target
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])

        # Extract features and target
        X = df[features]
        y = df[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluation
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)

        st.write("📊 Random Forest Model Evaluation")
        st.dataframe(pd.DataFrame(report).transpose())

        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax_cm)
        st.pyplot(fig_cm)

        # User input
        st.markdown("### 🔧 Enter Data for Prediction")
        user_input = {}
        for feature in features:
            user_input[feature] = st.number_input(
                f"Enter value for {feature.replace('_', ' ').title()}",
                value=float(df[feature].mean())
            )

        if st.button("Predict Accreditation"):
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            pred_label = le.inverse_transform([pred])[0]
            st.success(f"✅ Predicted Accreditation: **{pred_label}**")

    else:
        st.error("🚫 Required columns are not available in the dataset.")
