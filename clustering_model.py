import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ----------------------------
# Set Page Configuration
# ----------------------------
st.set_page_config(page_title="Clustering Analysis", layout="wide")

# Custom CSS สำหรับ UI ที่ดู clean และสวยงาม
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>
    body, p, h1, h2, h3, span {
        font-family: 'Poppins', sans-serif;
    }
    body {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #333;
    }
    .css-18e3th9, .block-container {
        background-color: #fff;
        padding: 2rem 3rem;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        background-color: #fff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    @media (prefers-color-scheme: dark) {
        body {
            background: linear-gradient(135deg, #232526, #414345);
            color: #f0f0f0;
        }
        .css-18e3th9, .block-container {
            background-color: #1e1e1e;
            box-shadow: 0 4px 8px rgba(255,255,255,0.1);
        }
        .css-1d391kg {
            background-color: #1e1e1e;
            box-shadow: 0 2px 4px rgba(255,255,255,0.1);
        }
        h1, h2, h3 {
            color: #f0f0f0;
        }
    }
    .stDownloadButton {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
    }
    .stDownloadButton:hover {
        background-color: #45a049;
    }
    button {
        font-family: 'Poppins', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ Clustering Analysis App")
st.write("Analyze and visualize clusters using KMeans and PCA with a clean, elegant UI.")

# ----------------------------
# Sidebar Settings
# ----------------------------
st.sidebar.header("⚙️ Settings")
st.sidebar.write("Customize clustering analysis options below.")
show_boxplot = st.sidebar.checkbox("Show Boxplot for Cluster Insights", value=True)
show_centroids = st.sidebar.checkbox("Show Cluster Centroids", value=True)
show_corr = st.sidebar.checkbox("Show Correlation Heatmap", value=True)
show_hist = st.sidebar.checkbox("Show Feature Histograms by Cluster", value=False)

# ----------------------------
# Function Definitions
# ----------------------------
@st.cache_data
def load_data(file):
    """โหลดข้อมูลจากไฟล์ CSV พร้อม caching เพื่อเพิ่มประสิทธิภาพ"""
    return pd.read_csv(file)

def compute_metrics(data_scaled, random_state=42):
    """
    คำนวณค่า Distortion (Inertia) และ Silhouette Score
    สำหรับ K = 2 ถึง 10
    """
    distortions = []
    silhouette_scores = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(data_scaled)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))
    return list(K_range), distortions, silhouette_scores

def plot_elbow_chart(K_range, distortions):
    fig, ax = plt.subplots()
    ax.plot(K_range, distortions, marker='o')
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Distortion (Inertia)")
    ax.set_title("Elbow Method for Optimal K")
    st.pyplot(fig)

def plot_silhouette_chart(K_range, silhouette_scores):
    fig, ax = plt.subplots()
    ax.plot(K_range, silhouette_scores, marker='o')
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score for Optimal K")
    st.pyplot(fig)

def plot_pca_scatter(data):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(x=data['PCA1'], y=data['PCA2'], hue=data['Cluster'],
                    palette='viridis', s=100, edgecolor='black', alpha=0.8, ax=ax)
    ax.set_xlabel("PCA Component 1", fontsize=12)
    ax.set_ylabel("PCA Component 2", fontsize=12)
    ax.set_title("Clusters Visualization using PCA", fontsize=14, weight='bold')
    st.pyplot(fig)

def plot_boxplot(data, feature):
    """
    สร้าง boxplot สำหรับ feature ที่เลือกในแต่ละ Cluster พร้อม annotate ค่า sum
    """
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x=data['Cluster'], y=data[feature], palette='viridis', ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel(feature)
    ax.set_title(f"Boxplot of {feature} by Cluster")

    # คำนวณค่า sum สำหรับแต่ละ Cluster
    sum_by_cluster = data.groupby("Cluster")[feature].sum()
    tick_positions = ax.get_xticks()
    for i, cluster in enumerate(sum_by_cluster.index):
        sum_val = sum_by_cluster.loc[cluster]
        max_val = data[data['Cluster'] == cluster][feature].max()
        ax.text(tick_positions[i], max_val, f"Sum: {sum_val:.2f}",
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=10, color='black', weight='bold')
    st.pyplot(fig)

def plot_corr_heatmap(data):
    """Plot correlation heatmap สำหรับข้อมูลทั้งหมด"""
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

def plot_feature_histograms(data, feature):
    """Plot histogram ของ feature ที่เลือกโดยแยกตาม Cluster"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x=feature, hue="Cluster", multiple="stack", palette="viridis", ax=ax)
    ax.set_title(f"Distribution of {feature} by Cluster")
    st.pyplot(fig)

def get_cluster_summary(data, numerical_features):
    """สรุปค่า mean, std, count และ sum สำหรับแต่ละ Cluster"""
    return data.groupby("Cluster")[numerical_features].agg(['mean', 'std', 'count', 'sum'])

def get_cluster_centroids(data, numerical_features):
    """คำนวณ centroids ของแต่ละ Cluster จากข้อมูลดิบ (ค่า mean)"""
    return data.groupby("Cluster")[numerical_features].mean()

# ----------------------------
# Main Application
# ----------------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"],
                                         help="Upload a CSV file with numerical data.")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("## 📊 Data Preview")
    st.dataframe(data.head(), use_container_width=True)
    
    # เลือกเฉพาะ columns ที่เป็น numerical
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numerical_features:
        st.error("❌ No numerical features found in the uploaded dataset.")
        st.stop()
    
    st.write("### 📌 Numerical Features Detected")
    st.write(numerical_features)
    
    # สร้างตัวแปร scaled สำหรับคำนวณ Metric (Elbow / Silhouette)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[numerical_features])
    
    # --- แสดง Elbow Method และ Silhouette Score สำหรับ K = 2..10 ---
    st.write("## 📉 Elbow Method & Silhouette Score")
    K_range, distortions, silhouette_scores = compute_metrics(data_scaled, random_state=32)
    col1, col2 = st.columns(2)
    with col1:
        plot_elbow_chart(K_range, distortions)
    with col2:
        plot_silhouette_chart(K_range, silhouette_scores)
    
    # --- ขั้นตอนการทำ Clustering แบบฟิกซ์ด้วย K=4 ---
    st.write("## 🚀 Perform Clustering with K=4 (Fixed)")

    kmeans = KMeans(
        n_clusters=4,
        init='k-means++',
        n_init='auto',
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=32,
        copy_x=True,
        algorithm='lloyd'
    )
    
    clusters = kmeans.fit_predict(data_scaled)
    data["Cluster"] = clusters
    
    # ทำ PCA เพื่อ Plot 2 มิติ
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    data['PCA1'] = pca_result[:, 0]
    data['PCA2'] = pca_result[:, 1]
    
    # แสดงผล PCA Scatter
    st.write("### Clusters Visualization (PCA 2D)")
    plot_pca_scatter(data)
    
    # คำนวณ Silhouette ของผลลัพธ์สุดท้าย
    silhouette_avg = silhouette_score(data_scaled, clusters)
    st.write(f"**Silhouette Score (K=4):** {silhouette_avg:.4f}")
    
    # --- สรุปข้อมูลของแต่ละคลัสเตอร์ ---
    st.write("## 📑 Cluster Summary (mean, std, count, sum)")
    cluster_summary = get_cluster_summary(data, numerical_features)
    st.dataframe(cluster_summary, use_container_width=True)
    
    # --- แสดง Centroid (Mean ของแต่ละคลัสเตอร์) ---
    if show_centroids:
        st.write("## 📋 Cluster Centroids")
        centroids = get_cluster_centroids(data, numerical_features)
        st.dataframe(centroids, use_container_width=True)
    
    # --- Heatmap ---
    if show_corr:
        st.write("## 🔥 Correlation Heatmap")
        plot_corr_heatmap(data[numerical_features])
    
    # --- Histogram แยกตามคลัสเตอร์ (ถ้าเลือก) ---
    if show_hist:
        selected_hist_feature = st.sidebar.selectbox("Select a feature for histogram:", numerical_features, index=0)
        st.write(f"## 📊 Distribution of {selected_hist_feature} by Cluster")
        plot_feature_histograms(data, selected_hist_feature)
    
    # --- Boxplot แยกตามคลัสเตอร์ (ถ้าเลือก) ---
    if show_boxplot:
        selected_boxplot_feature = st.sidebar.selectbox("Select feature for Boxplot", numerical_features, index=0)
        st.write(f"## 📊 Boxplot of {selected_boxplot_feature} by Cluster")
        plot_boxplot(data, selected_boxplot_feature)
    
    # --- ปุ่มดาวน์โหลดข้อมูลที่ใส่ Cluster แล้ว ---
    st.write("## 📥 Download Clustered Data")
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", data=csv, file_name="clustered_data.csv",
                       mime="text/csv",
                       help="Download the dataset with assigned clusters.")