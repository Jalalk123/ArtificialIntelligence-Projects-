import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors

st.set_page_config(
    page_title="Recipe Segmentation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{path}' was not found. Make sure the CSV is in the same folder as this script.")
        st.stop()

df_clustered = load_data('recipes_clustered_for_dashboard.csv')

df_clustered['Cluster'] = df_clustered['Cluster'].astype(str)

st.title("Tasty Bites - Recipe Segmentation Dashboard")
st.markdown("""
This interactive dashboard presents the results of the K-Means clustering analysis.
I have segmented recipes into **4 distinct groups** based on their nutritional characteristics, categories, and servings.
Use the filters in the sidebar to explore the data.
""")

st.sidebar.header("Visualization Options")

st.sidebar.subheader("Cluster Summary")
cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
st.sidebar.write("Number of Recipes per Cluster:")
st.sidebar.dataframe(cluster_counts.to_frame(name='Recipes'))

fig_counts = px.bar(
    cluster_counts,
    x=cluster_counts.index,
    y=cluster_counts.values,
    labels={'x': 'Cluster', 'y': 'Number of Recipes'},
    title='Distribution of Recipes by Cluster',
    color_discrete_sequence=plotly.colors.sequential.Viridis
)
st.plotly_chart(fig_counts, use_container_width=True)


st.header("Detailed Cluster Analysis")

all_clusters = sorted(df_clustered['Cluster'].unique())
selected_cluster_id = st.sidebar.selectbox(
    "Select a Cluster to view its characteristics:",
    options=['All'] + all_clusters,
    format_func=lambda x: f"Cluster {x}" if x != 'All' else 'All Clusters'
)

if selected_cluster_id == 'All':
    display_data = df_clustered
    chart_title_suffix = " (All Clusters)"
else:
    display_data = df_clustered[df_clustered['Cluster'] == selected_cluster_id]
    chart_title_suffix = f" (Cluster {selected_cluster_id})"

st.subheader(f"Average Nutritional Values{chart_title_suffix}")

if selected_cluster_id == 'All':
    avg_numeric_features = display_data[['calories', 'carbohydrate', 'sugar', 'protein', 'servings_numeric']].mean()
else:
    avg_numeric_features = display_data[['calories', 'carbohydrate', 'sugar', 'protein', 'servings_numeric']].mean()

df_avg_numeric = avg_numeric_features.reset_index()
df_avg_numeric.columns = ['Characteristic', 'Average Value']

fig_avg_nutrients = px.bar(
    df_avg_numeric,
    x='Characteristic',
    y='Average Value',
    title=f'Average Nutritional Values{chart_title_suffix}',
    color='Characteristic',
    color_discrete_sequence=px.colors.qualitative.Pastel
)
st.plotly_chart(fig_avg_nutrients, use_container_width=True)

st.subheader(f"Category Distribution{chart_title_suffix}")
category_counts = display_data['category'].value_counts(normalize=True).reset_index()
category_counts.columns = ['Category', 'Proportion']

fig_category = px.bar(
    category_counts,
    x='Category',
    y='Proportion',
    title=f'Proportion of Recipe Categories{chart_title_suffix}',
    color='Proportion',
    color_continuous_scale=px.colors.sequential.Viridis
)
st.plotly_chart(fig_category, use_container_width=True)

# --- High Traffic Distribution ---
st.subheader(f"High Traffic Distribution{chart_title_suffix}")
# Assuming the 'high_traffic' column (original or reconstructed) is available in the CSV
if 'high_traffic' in df_clustered.columns:
    traffic_counts = display_data['high_traffic'].value_counts(normalize=True).reset_index()
    traffic_counts.columns = ['Traffic Type', 'Proportion']
    
    fig_traffic = px.pie(
        traffic_counts,
        values='Proportion',
        names='Traffic Type',
        title=f'Proportion of Recipe Traffic{chart_title_suffix}',
        color_discrete_map={'High': 'darkgreen', 'No_Traffic_Info': 'gray', 'No_Traffic_Info_Original_NA': 'lightgray'}
    )
    st.plotly_chart(fig_traffic, use_container_width=True)
else:
    st.info("The original 'high_traffic' column was not found in the loaded data for this visualization.")
    # Fallback if only OHE columns are available
    if 'high_traffic_High' in df_clustered.columns:
        high_traffic_count = display_data['high_traffic_High'].sum()
        # Assuming 'No_Traffic_Info' is the other main category if 'High' is present
        # You might need to adjust this if your OHE columns are different (e.g., 'high_traffic_Low')
        no_traffic_info_count = display_data['high_traffic_No_Traffic_Info'].sum() if 'high_traffic_No_Traffic_Info' in df_clustered.columns else (len(display_data) - high_traffic_count)
        
        traffic_data = pd.DataFrame({
            'Traffic Type': ['High Traffic', 'No Traffic Info'],
            'Count': [high_traffic_count, no_traffic_info_count]
        })
        fig_traffic_ohe = px.pie(
            traffic_data,
            values='Count',
            names='Traffic Type',
            title=f'Proportion of Recipe Traffic{chart_title_suffix}',
            color_discrete_map={'High Traffic': 'darkgreen', 'No Traffic Info': 'gray'}
        )
        st.plotly_chart(fig_traffic_ohe, use_container_width=True)


st.header(" Cluster Visualization (PCA)")
st.markdown("""
This graph shows how the clusters are distributed in a reduced two-dimensional space
thanks to Principal Component Analysis (PCA).
""")

if 'PC1' in df_clustered.columns and 'PC2' in df_clustered.columns:
    fig_pca = px.scatter(
        df_clustered,
        x='PC1',
        y='PC2',
        color='Cluster',
        title='Recipe Clusters Reduced with PCA',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        hover_data=['calories', 'protein', 'category', 'servings_numeric'],
        color_discrete_map={str(i): c for i, c in zip(range(4), plotly.colors.sequential.Viridis)}
    )
    st.plotly_chart(fig_pca, use_container_width=True)
else:
    st.warning("PCA components (PC1, PC2) are not available in the loaded data file. To see this visualization, please ensure PCA components are saved in the CSV.")

    st.write("Unique values in high_traffic for current display_data:")
st.write(display_data['high_traffic'].value_counts())
