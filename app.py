import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="CORD-19 Research Explorer",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load the cleaned data."""
    try:
        df = pd.read_csv('cleaned_metadata.csv', low_memory=False)
        # Convert publish_year to int if it exists
        if 'publish_year' in df.columns:
            df['publish_year'] = df['publish_year'].astype('Int64')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def plot_publications_over_time(df):
    """Plot number of publications over time."""
    st.subheader("Publications Over Time")
    
    # Group by year and count publications
    yearly_counts = df['publish_year'].value_counts().sort_index()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(yearly_counts.index.astype(str), yearly_counts.values, color='skyblue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Publications')
    ax.set_title('Number of Publications by Year')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_top_journals(df, top_n=10):
    """Plot top publishing journals."""
    st.subheader(f"Top {top_n} Publishing Journals")
    
    # Get top journals
    top_journals = df['journal'].value_counts().head(top_n)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_journals.values, y=top_journals.index, palette='viridis')
    ax.set_xlabel('Number of Publications')
    ax.set_ylabel('Journal')
    ax.set_title(f'Top {top_n} Journals by Publication Count')
    st.pyplot(fig)

def generate_wordcloud(df):
    """Generate a word cloud from paper titles."""
    st.subheader("Common Words in Paper Titles")
    
    # Combine all titles
    text = ' '.join(title for title in df['title'].astype(str))
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Display the word cloud
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def main():
    st.title("📚 CORD-19 Research Explorer")
    st.markdown("""
    This interactive dashboard explores the CORD-19 dataset, which contains metadata 
    about COVID-19 research papers. Use the sidebar to filter the data and explore 
    different aspects of the dataset.
    """)
    
    # Load the data
    df = load_data()
    
    if df is not None:
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Year range slider
        if 'publish_year' in df.columns:
            min_year = int(df['publish_year'].min())
            max_year = int(df['publish_year'].max())
            year_range = st.sidebar.slider(
                "Select Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
            
            # Filter data based on year range
            df = df[(df['publish_year'] >= year_range[0]) & 
                   (df['publish_year'] <= year_range[1])]
        
        # Journal selection
        if 'journal' in df.columns:
            journals = ['All'] + sorted(df['journal'].dropna().unique().tolist())
            selected_journal = st.sidebar.selectbox(
                "Select Journal",
                journals,
                index=0
            )
            
            if selected_journal != 'All':
                df = df[df['journal'] == selected_journal]
        
        # Display dataset info
        st.sidebar.markdown("---")
        st.sidebar.info(f"**Dataset Info:**\n"
                       f"- Total papers: {len(df):,}\n"
                       f"- Time period: {min_year} - {max_year}")
        
        # Main content
        col1, col2 = st.columns([3, 1])
        
        with col1:
            plot_publications_over_time(df)
            
            # Show top journals only if we're not filtering by journal
            if 'journal' in df.columns and (selected_journal == 'All' or len(df) > 0):
                plot_top_journals(df)
            
            if not df.empty:
                generate_wordcloud(df)
        
        with col2:
            st.subheader("Dataset Sample")
            st.dataframe(df[['title', 'journal', 'publish_year']].head(10))
            
            # Download button for filtered data
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name=f"cord19_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
                help="Download the filtered dataset as a CSV file"
            )
        
        # Show data summary
        st.subheader("Data Summary")
        st.json({
            "Total Papers": len(df),
            "Unique Journals": df['journal'].nunique(),
            "Earliest Publication Year": int(df['publish_year'].min()) if 'publish_year' in df.columns else None,
            "Latest Publication Year": int(df['publish_year'].max()) if 'publish_year' in df.columns else None,
            "Average Title Length (words)": round(df['title'].str.split().str.len().mean(), 1)
        })

if __name__ == "__main__":
    main()
