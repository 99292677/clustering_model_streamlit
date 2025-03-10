import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# -------------------------------------------------------------
# Set Page Configuration & Custom CSS
# -------------------------------------------------------------
st.set_page_config(page_title="Clustering Analysis", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>
    body, p, h1, h2, h3, span {
        font-family: 'Poppins', sans-serif;
    }
    body {
        background: linear-gradient(135deg, #f7f7f7, #ffffff);
        color: #333;
        margin: 0;
        padding: 0;
    }
    .css-18e3th9, .block-container {
        background-color: #fff;
        padding: 2rem 3rem;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .css-1d391kg {
        background-color: #fff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    @media (prefers-color-scheme: dark) {
        body {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: #f0f0f0;
        }
        .css-18e3th9, .block-container {
            background-color: #2c3e50;
            box-shadow: 0 4px 8px rgba(255,255,255,0.1);
        }
        .css-1d391kg {
            background-color: #2c3e50;
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
    .my-radio-label {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Sidebar Navigation (6 Pages)
# -------------------------------------------------------------
page = st.sidebar.radio(
    "Select a Page",
    ("Clustering", "Insights", "Cluster Heatmaps", "Cluster Boxplots", "Summary Stats", "Barplots"),
    index=0,
    help="Choose which page to display"
)
st.sidebar.write("---")

# -------------------------------------------------------------
# st.session_state: Store DataFrame and related variables
# -------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "data_scaled" not in st.session_state:
    st.session_state.data_scaled = None
if "numerical_features" not in st.session_state:
    st.session_state.numerical_features = None

# -------------------------------------------------------------
# Define Color Palettes
# -------------------------------------------------------------
# สำหรับ Plot แบบ discrete ใช้ "Paired"
paired_pal = sns.color_palette("Paired", n_colors=10)
# สำหรับ Heatmap และตาราง ใช้ "Blues"
blues_cmap = sns.color_palette("Blues", as_cmap=True)

# -------------------------------------------------------------
# Shared Functions
# -------------------------------------------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def compute_metrics(data_scaled, random_state=42):
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
    fig, ax = plt.subplots(figsize=(12,8), dpi=120)
    ax.plot(K_range, distortions, marker='o', color='black')
    ax.set_xlabel("Number of Clusters (K)", fontsize=14)
    ax.set_ylabel("Distortion (Inertia)", fontsize=14)
    ax.set_title("Elbow Method for Optimal K", fontsize=16, weight='bold')
    plt.tight_layout()
    st.pyplot(fig)

def plot_silhouette_chart(K_range, silhouette_scores):
    fig, ax = plt.subplots(figsize=(12,8), dpi=120)
    ax.plot(K_range, silhouette_scores, marker='o', color='black')
    ax.set_xlabel("Number of Clusters (K)", fontsize=14)
    ax.set_ylabel("Silhouette Score", fontsize=14)
    ax.set_title("Silhouette Score for Optimal K", fontsize=16, weight='bold')
    plt.tight_layout()
    st.pyplot(fig)

def plot_pca_scatter(data, ax):
    sns.scatterplot(
        x='PCA1', y='PCA2',
        hue='Cluster',
        data=data,
        palette=paired_pal,
        s=120, edgecolor='black', alpha=0.85,
        ax=ax
    )
    ax.set_xlabel("PCA Component 1", fontsize=14)
    ax.set_ylabel("PCA Component 2", fontsize=14)
    ax.set_title("PCA Scatter Plot (2D)", fontsize=16, weight='bold')

def plot_tsne_scatter(data, ax):
    sns.scatterplot(
        x='tSNE1', y='tSNE2',
        hue='Cluster',
        data=data,
        palette=paired_pal,
        s=120, edgecolor='black', alpha=0.85,
        ax=ax
    )
    ax.set_xlabel("t-SNE Component 1", fontsize=14)
    ax.set_ylabel("t-SNE Component 2", fontsize=14)
    ax.set_title("t-SNE Scatter Plot (2D)", fontsize=16, weight='bold')

def plot_feature_histograms(data, feature):
    fig, ax = plt.subplots(figsize=(12,8), dpi=120)
    sns.histplot(data=data, x=feature, hue="Cluster", multiple="stack", palette=paired_pal, ax=ax)
    ax.set_xlabel(feature, fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(f"Distribution of {feature} by Cluster", fontsize=16, weight='bold')
    plt.tight_layout()
    st.pyplot(fig)

def plot_boxplot(data, feature):
    data['Cluster'] = data['Cluster'].astype(str)
    fig, ax = plt.subplots(figsize=(12,8), dpi=120)
    sns.boxplot(x=feature, y='Cluster', data=data, palette=paired_pal, ax=ax)
    ax.set_xlabel(feature, fontsize=14)
    ax.set_ylabel("Cluster", fontsize=14)
    ax.set_title(f"Boxplot of {feature} by Cluster (Horizontal)", fontsize=16, weight='bold')
    sum_by_cluster = data.groupby("Cluster")[feature].sum()
    tick_positions = ax.get_yticks()
    for i, cluster_label in enumerate(sum_by_cluster.index):
        sum_val = sum_by_cluster.loc[cluster_label]
        max_val = data[data['Cluster'] == cluster_label][feature].max()
        ax.text(x=max_val, y=tick_positions[i], s=f"Sum: {sum_val:,.2f}",
                verticalalignment='center', horizontalalignment='left',
                fontsize=10, color='black', weight='bold')
    plt.tight_layout()
    st.pyplot(fig)

def plot_corr_heatmap(data):
    numeric_cols = data.select_dtypes(include=[np.number])
    if numeric_cols.shape[1] < 2:
        st.warning("Not enough numeric columns to compute correlation (all data).")
        return
    corr = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(12,8), dpi=120)
    sns.heatmap(corr, annot=True, cmap=blues_cmap, ax=ax, vmin=-1, vmax=1, fmt=",.2f")
    ax.set_title("Correlation Heatmap (All Data)", fontsize=16, weight='bold')
    plt.tight_layout()
    st.pyplot(fig)

def plot_corr_heatmap_per_cluster(data, cluster_label):
    subset = data[data["Cluster"] == cluster_label]
    numeric_subset = subset.select_dtypes(include=[np.number])
    if numeric_subset.shape[1] < 2:
        st.warning(f"Not enough numeric columns in Cluster {cluster_label} to compute correlation.")
        return
    if len(numeric_subset) < 2:
        st.warning(f"Not enough data in Cluster {cluster_label} to compute correlation.")
        return
    corr = numeric_subset.corr()
    fig, ax = plt.subplots(figsize=(12,8), dpi=120)
    sns.heatmap(corr, annot=True, cmap=blues_cmap, ax=ax, vmin=-1, vmax=1, fmt=",.2f")
    ax.set_title(f"Correlation Heatmap (Cluster {cluster_label})", fontsize=16, weight='bold')
    plt.tight_layout()
    st.pyplot(fig)

def get_cluster_summary(data, numerical_features):
    return data.groupby("Cluster")[numerical_features].agg(['mean', 'std', 'count', 'sum'])

def get_summary_statistics_by_cluster(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    grouped_desc = data.groupby("Cluster")[numeric_cols].describe()
    return grouped_desc

def get_cluster_centroids(data, numerical_features):
    return data.groupby("Cluster")[numerical_features].mean()

# -------------------------------------------------------------
# New: Custom Summary Stats (Vertical) for Specific Columns with Styling
# -------------------------------------------------------------
def get_custom_summary_by_cluster_vertical(data):
    """
    สร้างตาราง describe สำหรับแต่ละคอลัมน์ที่ระบุในแนวตั้ง
    แต่ละคอลัมน์จะแสดงตาราง describe แยกกัน
    ใช้ Styler เพื่อเพิ่ม background gradient ด้วย cmap 'Blues'
    พร้อมใช้ .format("{:,.2f}") เพื่อให้ตัวเลขมี comma
    """
    columns_of_interest = [
        'Total Service Amount',
        'Service amount Per Unique Quotations',
        'Sum Unique Quotations',
        'Avg Period Month',
        'Count Acc.Name',
        'AVG Active Amount per Month'
    ]
    columns_existing = [col for col in columns_of_interest if col in data.columns]
    if not columns_existing:
        return pd.DataFrame({"Warning": ["No matched columns found in dataset."]})
    
    summary_tables = {}
    for col in columns_existing:
        desc = data.groupby("Cluster")[col].describe()
        desc_t = desc.transpose()
        styled_table = desc_t.style.format("{:,.2f}").background_gradient(cmap=blues_cmap, low=0.2, high=0.8)
        summary_tables[col] = styled_table
    return summary_tables

# -------------------------------------------------------------
# New: Function to Plot Barplot for a Variable by Cluster with Annotations
# -------------------------------------------------------------
def plot_barplot(data, variable):
    # คำนวณค่าเฉลี่ยของแต่ละ Cluster สำหรับตัวแปรนี้
    cluster_mean = data.groupby("Cluster")[variable].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10,6), dpi=120)
    barplot = sns.barplot(x="Cluster", y=variable, data=cluster_mean, palette=paired_pal, ax=ax)
    ax.set_xlabel("Cluster", fontsize=14)
    ax.set_ylabel(variable, fontsize=14)
    ax.set_title(f"Barplot of {variable} by Cluster", fontsize=16, weight='bold')
    # เพิ่ม Annotation: แสดงค่าเฉลี่ยบนแต่ละบาร์ พร้อม comma formatting
    for p in barplot.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.01*height, f'{height:,.2f}', 
                ha="center", va="bottom", fontsize=12, color='black')
    plt.tight_layout()
    st.pyplot(fig)

# -------------------------------------------------------------
# File Uploader
# -------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"], help="Upload a CSV file with numerical data.")
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.session_state.df = data
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    st.session_state.numerical_features = numerical_features
    if not numerical_features:
        st.error("❌ No numerical features found in the uploaded dataset.")
        st.stop()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[numerical_features])
    st.session_state.data_scaled = data_scaled

if st.session_state.df is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

data = st.session_state.df.copy()
data_scaled = st.session_state.data_scaled
numerical_features = st.session_state.numerical_features

# -------------------------------------------------------------
# Page: Clustering
# -------------------------------------------------------------
if page == "Clustering":
    st.title("✨ Clustering Analysis")
    st.write("Analyze and visualize clusters using KMeans, PCA, and t-SNE.")
    st.write("### Data Preview")
    st.dataframe(data.head(), use_container_width=True)
    st.write("### Numerical Features Detected")
    st.write(numerical_features)
    st.write("## Elbow Method & Silhouette Score")
    K_range, distortions, silhouette_scores = compute_metrics(data_scaled, random_state=32)
    col1, col2 = st.columns(2)
    with col1:
        plot_elbow_chart(K_range, distortions)
    with col2:
        plot_silhouette_chart(K_range, silhouette_scores)
    k_value = st.slider("Select the number of clusters (K)", min_value=2, max_value=10, value=4, step=1)
    if st.button("Run Clustering"):
        kmeans = KMeans(
            n_clusters=k_value,
            init='k-means++',
            n_init='auto',
            max_iter=300,
            tol=1e-4,
            random_state=32,
            algorithm='lloyd'
        )
        clusters = kmeans.fit_predict(data_scaled)
        data["Cluster"] = clusters
        pca = PCA(n_components=2, random_state=32)
        pca_result = pca.fit_transform(data_scaled)
        data['PCA1'] = pca_result[:, 0]
        data['PCA2'] = pca_result[:, 1]
        tsne = TSNE(n_components=2, random_state=32)
        tsne_result = tsne.fit_transform(data_scaled)
        data["tSNE1"] = tsne_result[:, 0]
        data["tSNE2"] = tsne_result[:, 1]
        st.session_state.df = data
        st.success(f"Clustering with K={k_value} completed!")
        st.write("### Clusters Visualization (PCA vs t-SNE)")
        col_left, col_right = st.columns(2)
        with col_left:
            fig1, ax1 = plt.subplots(figsize=(12,8), dpi=120)
            plot_pca_scatter(data, ax1)
            plt.tight_layout()
            st.pyplot(fig1)
        with col_right:
            fig2, ax2 = plt.subplots(figsize=(12,8), dpi=120)
            plot_tsne_scatter(data, ax2)
            plt.tight_layout()
            st.pyplot(fig2)
        silhouette_avg = silhouette_score(data_scaled, clusters)
        st.write(f"**Silhouette Score (K={k_value}):** {silhouette_avg:.4f}")
        st.write("## Cluster Summary (mean, std, count, sum)")
        cluster_summary = get_cluster_summary(data, numerical_features)
        st.dataframe(cluster_summary.style.format("{:,.2f}"), use_container_width=True)
        st.write("## Download Clustered Data")
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="clustered_data.csv", mime="text/csv", help="Download the dataset with assigned clusters.")
    else:
        st.info("Click 'Run Clustering' to perform KMeans and see results.")

# -------------------------------------------------------------
# Page: Insights
# -------------------------------------------------------------
elif page == "Insights":
    st.title("📊 Insights & Visualizations")
    st.write("Further analysis: Histograms, Centroids, Correlation Heatmap, and Cluster Characteristics Heatmap.")
    if "Cluster" not in data.columns:
        st.info("No clusters found yet. Running default K=4 clustering.")
        kmeans_default = KMeans(
            n_clusters=4,
            init='k-means++',
            n_init='auto',
            max_iter=300,
            tol=1e-4,
            random_state=32,
            algorithm='lloyd'
        )
        clusters_default = kmeans_default.fit_predict(data_scaled)
        data["Cluster"] = clusters_default
        pca = PCA(n_components=2, random_state=32)
        pca_result = pca.fit_transform(data_scaled)
        data['PCA1'] = pca_result[:, 0]
        data['PCA2'] = pca_result[:, 1]
        tsne = TSNE(n_components=2, random_state=32)
        tsne_result = tsne.fit_transform(data_scaled)
        data["tSNE1"] = tsne_result[:, 0]
        data["tSNE2"] = tsne_result[:, 1]
        st.session_state.df = data
    show_hist = st.checkbox("Show Feature Histograms by Cluster", value=True)
    show_centroids = st.checkbox("Show Cluster Centroids", value=True)
    show_corr_all = st.checkbox("Show Correlation Heatmap (All Data)", value=True)
    show_cluster_chars = st.checkbox("Show Cluster Characteristics Heatmap", value=True)
    selected_feature = st.selectbox("Select a feature to visualize by Cluster", numerical_features, index=0)
    if show_hist:
        st.write(f"### Distribution of {selected_feature} by Cluster")
        plot_feature_histograms(data, selected_feature)
    if show_centroids:
        st.write("### Cluster Centroids")
        centroids = get_cluster_centroids(data, numerical_features)
        st.dataframe(centroids.style.format("{:,.2f}"), use_container_width=True)
    if show_corr_all:
        st.write("### Correlation Heatmap (All Data)")
        plot_corr_heatmap(data)
    if show_cluster_chars:
        st.write("### Heatmap of Cluster Characteristics")
        variables = [
            'Total Service Amount',
            'Service amount Per Unique Quotations',
            'Sum Unique Quotations',
            'Avg Period Month',
            'Count Acc.Name',
            'AVG Active Amount per Month'
        ]
        existing_vars = [v for v in variables if v in data.columns]
        if not existing_vars:
            st.warning("None of the selected variables are available in the dataset.")
        else:
            cluster_analysis = data.groupby("Cluster")[existing_vars].mean()
            fig, ax = plt.subplots(figsize=(10,6), dpi=120)
            sns.heatmap(cluster_analysis.T, cmap="coolwarm", annot=True, fmt=",.2f", ax=ax)
            ax.set_xlabel("Cluster", fontsize=14)
            ax.set_ylabel("Variable", fontsize=14)
            ax.set_title("Heatmap of Cluster Characteristics", fontsize=16, weight='bold')
            plt.tight_layout()
            st.pyplot(fig)

# -------------------------------------------------------------
# Page: Cluster Heatmaps
# -------------------------------------------------------------
elif page == "Cluster Heatmaps":
    st.title("🔥 Correlation Heatmaps per Cluster")
    st.write("View correlation heatmaps separately for each cluster.")
    if "Cluster" not in data.columns:
        st.info("No clusters found yet. Running default K=4 clustering.")
        kmeans_default = KMeans(
            n_clusters=4,
            init='k-means++',
            n_init='auto',
            max_iter=300,
            tol=1e-4,
            random_state=32,
            algorithm='lloyd'
        )
        clusters_default = kmeans_default.fit_predict(data_scaled)
        data["Cluster"] = clusters_default
        pca = PCA(n_components=2, random_state=32)
        pca_result = pca.fit_transform(data_scaled)
        data['PCA1'] = pca_result[:, 0]
        data['PCA2'] = pca_result[:, 1]
        tsne = TSNE(n_components=2, random_state=32)
        tsne_result = tsne.fit_transform(data_scaled)
        data["tSNE1"] = tsne_result[:, 0]
        data["tSNE2"] = tsne_result[:, 1]
        st.session_state.df = data
    st.write("Below are correlation heatmaps for each cluster:")
    unique_clusters = sorted(data["Cluster"].unique())
    for c in unique_clusters:
        st.write(f"## Cluster {c}")
        plot_corr_heatmap_per_cluster(data, cluster_label=c)

# -------------------------------------------------------------
# Page: Cluster Boxplots
# -------------------------------------------------------------
elif page == "Cluster Boxplots":
    st.title("📦 Boxplots of All Numeric Features")
    st.write("View boxplots of every numeric feature by cluster all at once.")
    if "Cluster" not in data.columns:
        st.info("No clusters found yet. Running default K=4 clustering.")
        kmeans_default = KMeans(
            n_clusters=4,
            init='k-means++',
            n_init='auto',
            max_iter=300,
            tol=1e-4,
            random_state=32,
            algorithm='lloyd'
        )
        clusters_default = kmeans_default.fit_predict(data_scaled)
        data["Cluster"] = clusters_default
        pca = PCA(n_components=2, random_state=32)
        pca_result = pca.fit_transform(data_scaled)
        data['PCA1'] = pca_result[:, 0]
        data['PCA2'] = pca_result[:, 1]
        tsne = TSNE(n_components=2, random_state=32)
        tsne_result = tsne.fit_transform(data_scaled)
        data["tSNE1"] = tsne_result[:, 0]
        data["tSNE2"] = tsne_result[:, 1]
        st.session_state.df = data
    for feature in numerical_features:
        st.write(f"## Boxplot of {feature} by Cluster")
        plot_boxplot(data, feature)

# -------------------------------------------------------------
# Page: Summary Stats
# -------------------------------------------------------------
elif page == "Summary Stats":
    st.title("📊 Summary Statistics by Cluster")
    st.write("View custom summary statistics for selected columns grouped by cluster in a vertical format.")
    if "Cluster" not in data.columns:
        st.info("No clusters found yet. Running default K=4 clustering.")
        kmeans_default = KMeans(
            n_clusters=4,
            init='k-means++',
            n_init='auto',
            max_iter=300,
            tol=1e-4,
            random_state=32,
            algorithm='lloyd'
        )
        clusters_default = kmeans_default.fit_predict(data_scaled)
        data["Cluster"] = clusters_default
        pca = PCA(n_components=2, random_state=32)
        pca_result = pca.fit_transform(data_scaled)
        data['PCA1'] = pca_result[:, 0]
        data['PCA2'] = pca_result[:, 1]
        tsne = TSNE(n_components=2, random_state=32)
        tsne_result = tsne.fit_transform(data_scaled)
        data["tSNE1"] = tsne_result[:, 0]
        data["tSNE2"] = tsne_result[:, 1]
        st.session_state.df = data
    st.write("### Custom Summary Statistics (Vertical) by Cluster")
    columns_of_interest = [
        'Total Service Amount',
        'Service amount Per Unique Quotations',
        'Sum Unique Quotations',
        'Avg Period Month',
        'Count Acc.Name',
        'AVG Active Amount per Month'
    ]
    columns_existing = [col for col in columns_of_interest if col in data.columns]
    if not columns_existing:
        st.error("No matching columns found for custom summary.")
    else:
        for col in columns_existing:
            st.write(f"#### {col}")
            custom_desc = data.groupby("Cluster")[col].describe().transpose()
            styled_table = custom_desc.style.format("{:,.2f}").background_gradient(cmap=blues_cmap, low=0.2, high=0.8)
            st.write(styled_table)

# -------------------------------------------------------------
# Page: Barplots
# -------------------------------------------------------------
elif page == "Barplots":
    st.title("📊 Barplots of Cluster Characteristics")
    st.write("View barplots for selected variables by cluster, with annotated values.")
    if "Cluster" not in data.columns:
        st.info("No clusters found yet. Running default K=4 clustering.")
        kmeans_default = KMeans(
            n_clusters=4,
            init='k-means++',
            n_init='auto',
            max_iter=300,
            tol=1e-4,
            random_state=32,
            algorithm='lloyd'
        )
        clusters_default = kmeans_default.fit_predict(data_scaled)
        data["Cluster"] = clusters_default
        st.session_state.df = data

    variables = [
        'Total Service Amount',
        'Service amount Per Unique Quotations',
        'Sum Unique Quotations',
        'Avg Period Month',
        'Count Acc.Name',
        'AVG Active Amount per Month'
    ]
    existing_vars = [var for var in variables if var in data.columns]
    if not existing_vars:
        st.error("None of the selected variables are available in the dataset.")
    else:
        for var in existing_vars:
            st.write(f"#### Barplot of {var} by Cluster")
            plot_barplot(data, var)
