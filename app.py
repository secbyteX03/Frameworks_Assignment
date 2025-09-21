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
    """Plot number of publications over time with enhanced styling and interactivity."""
    st.subheader("📅 Publications Over Time")
    
    # Group by year and count publications
    yearly_counts = df['publish_year'].value_counts().sort_index()
    
    # Create the plot with enhanced styling
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use a more appealing color palette
    colors = plt.cm.viridis(range(len(yearly_counts)))
    
    # Create bar plot with improved styling
    bars = ax.bar(
        yearly_counts.index.astype(str), 
        yearly_counts.values, 
        color=colors,
        edgecolor='white',
        linewidth=0.7,
        alpha=0.8
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + (0.02 * max(yearly_counts.values)),
            f'{int(height):,}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # Customize axes and title
    ax.set_xlabel('Year', fontsize=12, labelpad=10)
    ax.set_ylabel('Number of Publications', fontsize=12, labelpad=10)
    ax.set_title('📈 Publication Trends Over Time', fontsize=14, pad=15, fontweight='bold')
    
    # Improve grid and layout
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Add a brief interpretation
    st.caption("This chart shows the distribution of publications across different years. "
              "Hover over the bars to see exact publication counts.")

def plot_top_journals(df, top_n=10):
    """Plot top publishing journals with enhanced styling and interactivity."""
    st.subheader(f"🏆 Top {top_n} Publishing Journals")
    
    # Add a slider to adjust the number of journals shown
    top_n = st.slider(
        "Number of top journals to show:",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
        key="top_n_journals"
    )
    
    # Get top journals
    top_journals = df['journal'].value_counts().head(top_n)
    
    # Calculate publication percentages
    total_papers = len(df)
    top_journals_pct = (top_journals / total_papers * 100).round(1)
    
    # Create a color gradient based on the rank
    colors = plt.cm.viridis_r(range(top_n))
    
    # Create the plot with enhanced styling
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar plot
    bars = ax.barh(
        top_journals.index.astype(str), 
        top_journals.values,
        color=colors,
        edgecolor='white',
        linewidth=0.7
    )
    
    # Add value labels on the bars
    for i, (value, pct) in enumerate(zip(top_journals.values, top_journals_pct)):
        ax.text(
            value + (0.01 * max(top_journals.values)),  # x position
            i,  # y position
            f"{value:,} ({pct}%)",
            va='center',
            ha='left',
            fontsize=10
        )
    
    # Customize the plot
    ax.set_xlabel('Number of Publications', fontsize=12, labelpad=10)
    ax.set_ylabel('Journal', fontsize=12, labelpad=10)
    ax.set_title(f'Top {top_n} Journals by Publication Count', 
                fontsize=14, pad=15, fontweight='bold')
    
    # Improve layout and grid
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Add a divider and some context
    st.pyplot(fig)
    st.caption(f"""
    This chart shows the top {top_n} journals by publication count in the dataset.
    The percentage represents the proportion of papers from each journal relative to the total number of papers.
    """)

def generate_wordcloud(df):
    """Generate an enhanced word cloud from paper titles with better styling and interactivity."""
    st.subheader("🔠 Word Cloud of Paper Titles")
    
    # Add a slider for the number of words
    max_words = st.slider(
        "Select maximum number of words:",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Adjust the number of words shown in the word cloud"
    )
    
    # Add a color scheme selector
    color_scheme = st.selectbox(
        "Select color scheme:",
        ["viridis", "plasma", "inferno", "magma", "cividis"],
        index=0,
        help="Choose a color scheme for the word cloud"
    )
    
    # Combine all titles and preprocess
    text = ' '.join(title for title in df['title'].astype(str))
    
    # Generate word cloud with enhanced settings
    wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color='white',
        max_words=max_words,
        colormap=color_scheme,
        contour_width=1,
        contour_color='steelblue',
        collocations=False,  # Don't include common word pairs
        min_font_size=8,
        max_font_size=150,
        random_state=42
    ).generate(text)
    
    # Display the word cloud with improved styling
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(wordcloud, interpolation='bilinear', aspect='auto')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Add a title and description
    st.pyplot(fig)
    st.caption("""
    This word cloud visualizes the most common terms in paper titles. 
    Larger words appear more frequently. Use the controls above to adjust the visualization.
    """)

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
