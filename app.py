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

# Title
st.title("🎓 Prediction & Analysis of PDDIKTI Study Programs")

# Load and clean data
df = pd.read_csv("dataset_pddikti_bersih.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select Page", ["📊 Dataset & Visualization", "🔍 Clustering", "🧠 Accreditation Prediction"])

# Features and Target
features = ['jumlah_semester', 'jumlah_mk', 'jumlah_sks', 'total_sks', 'mahasiswa_aktif']
target = 'akreditasi'

# Check column validity
if not all(col in df.columns for col in features + [target]):
    st.error("❌ Required columns not found in the dataset.")
    st.stop()

# Encode target
le = LabelEncoder()
df[target] = le.fit_transform(df[target])

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ==========================
# Page 1: Dataset & Visualization
# ==========================
if page == "📊 Dataset & Visualization":
    st.subheader("📋 PDDIKTI Dataset")
    st.dataframe(df)

    st.subheader("📌 Accreditation Distribution")
    fig1, ax1 = plt.subplots()
    df['akreditasi'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_xlabel("Accreditation (Encoded)")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    st.subheader("📈 Feature Correlation")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df[features + [target]].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

# ==========================
# Page 2: Clustering
# ==========================
elif page == "🔍 Clustering":
    st.subheader("🔎 KMeans Clustering")
    k = st.slider("Select number of clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # PCA for dimensionality reduction to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['cluster'] = clusters

    fig3, ax3 = plt.subplots()
    scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    ax3.set_title("Clustering Visualization (PCA)")
    st.pyplot(fig3)

    st.write("Number of data points per cluster:")
    st.write(df['cluster'].value_counts())

# ==========================
# Page 3: Accreditation Prediction
# ==========================
elif page == "🧠 Accreditation Prediction":
    st.subheader("📌 Predict Study Program Accreditation")

    # Split data
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.write("📊 Model Evaluation:")
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("🧩 Confusion Matrix")
    fig4, ax4 = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax4)
    st.pyplot(fig4)

    # User input
    st.subheader("📝 Try Predicting Yourself")

    user_input = {}
    for feature in features:
        val = st.number_input(f"Enter value for {feature}", value=float(df[feature].mean()))
        user_input[feature] = val

    if st.button("Predict Accreditation"):
        input_df = pd.DataFrame([user_input])
        pred = model.predict(input_df)[0]
        pred_label = le.inverse_transform([pred])[0]
        st.success(f"✅ Predicted Accreditation: **{pred_label}**")
