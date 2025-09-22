import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime
import matplotlib as mpl

# Configure matplotlib to use a font that supports emojis
plt.rcParams['font.sans-serif'] = ['Segoe UI Emoji', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # For minus sign display

# Set page config
st.set_page_config(
    page_title="CORD-19 Research Explorer",
    page_icon="📊",
    layout="wide"
)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Load the cleaned data with memory optimization."""
    try:
        # Define column data types to optimize memory usage
        dtypes = {
            'publish_year': 'float32',
            'title_word_count': 'int16',
            # Add other columns with appropriate dtypes if known
        }
        
        # Only read necessary columns to save memory
        use_cols = ['title', 'journal', 'publish_year', 'title_word_count']
        
        # Read data in chunks if needed
        chunks = []
        for chunk in pd.read_csv('cleaned_metadata.csv', 
                               low_memory=False, 
                               dtype=dtypes, 
                               usecols=use_cols,
                               chunksize=10000):  # Process in chunks of 10,000 rows
            chunks.append(chunk)
        
        # Combine chunks
        df = pd.concat(chunks, axis=0)
        
        # Convert year to Int64 (nullable integer type)
        if 'publish_year' in df.columns:
            df['publish_year'] = df['publish_year'].fillna(-1).astype('int32')
            df['publish_year'] = df['publish_year'].replace(-1, pd.NA).astype('Int64')
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Please ensure the 'cleaned_metadata.csv' file exists in the same directory as the app.")
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
            height + (0.02 * max(yearly_counts.values)) if not yearly_counts.empty else 0,
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
    
    # Add interactive hover functionality with better error handling
    try:
        import mplcursors
        
        # Create a custom hover function
        def on_hover(sel):
            try:
                idx = sel.target.index
                if isinstance(idx, (int, np.integer)) and 0 <= idx < len(yearly_counts):
                    year = yearly_counts.index[idx]
                    count = yearly_counts.iloc[idx]
                    sel.annotation.set_text(f"Year: {year}\nPublications: {int(count):,}")
                    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9, boxstyle="round,pad=0.5")
            except Exception as e:
                print(f"Error in hover: {e}")
        
        # Configure the cursor
        cursor = mplcursors.cursor(bars, hover=True)
        cursor.connect("add", on_hover)
        
    except ImportError:
        st.warning("mplcursors not installed. Hover functionality will be limited.")
        st.warning("Install it with: pip install mplcursors")
    
    # Display the plot in Streamlit with tight layout
    st.pyplot(fig, clear_figure=True)
    
    # Add a brief interpretation
    st.caption("""
    This chart shows the distribution of publications across different years. 
    Hover over the bars to see exact publication counts.
    """)

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

def preprocess_text(text):
    """Preprocess text for word cloud generation."""
    if not isinstance(text, str):
        return ''
    # Convert to lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    # Remove common stopwords and short words
    stopwords = set(['the', 'and', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'from', 'by', 'as', 'an', 'at', 'is', 'are', 'was', 'were', 'be', 'this', 'that', 'these', 'those'])
    words = [word for word in text.split() if len(word) > 2 and word not in stopwords]
    return ' '.join(words)

def generate_wordcloud(df):
    """Generate an enhanced word cloud from paper titles with better memory efficiency."""
    st.subheader("Word Cloud of Paper Titles")
    
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
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Processing titles... (this may take a moment)")
    
    try:
        # Process in chunks to save memory
        chunk_size = 1000
        processed_texts = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            # Process titles in the current chunk
            chunk_text = ' '.join(chunk['title'].fillna('').astype(str).apply(preprocess_text))
            processed_texts.append(chunk_text)
            # Update progress
            progress = min((i + len(chunk)) / len(df), 1.0)
            progress_bar.progress(progress)
        
        # Combine all processed chunks
        all_text = ' '.join(processed_texts)
        
        # If text is empty, show a message and return
        if not all_text.strip():
            st.warning("No valid text data found for generating word cloud.")
            return
        
        status_text.text("Generating word cloud...")
        
        # Generate word cloud with optimized settings
        wordcloud = WordCloud(
            width=1000,
            height=600,
            background_color='white',
            max_words=max_words,
            colormap=color_scheme,
            contour_width=1,
            contour_color='steelblue',
            collocations=False,
            min_font_size=8,
            max_font_size=150,
            random_state=42,
            prefer_horizontal=0.9,  # More horizontal text for better readability
            scale=2  # Higher scale for better quality
        ).generate(all_text)
        
        # Display the word cloud
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(wordcloud, interpolation='bilinear', aspect='auto')
        ax.axis('off')
        
        # Display the plot
        st.pyplot(fig, bbox_inches='tight', pad_inches=0)
        
        # Add a caption
        st.caption("""
        This word cloud visualizes the most common terms in paper titles. 
        Larger words appear more frequently. Use the controls above to adjust the visualization.
        """)
        
    except Exception as e:
        st.error(f"An error occurred while generating the word cloud: {str(e)}")
    finally:
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

def main():
    st.title("📚 CORD-19 Research Explorer")
    st.markdown("""
    This interactive dashboard explores the CORD-19 dataset, which contains metadata 
    about COVID-19 research papers. Use the sidebar to filter the data and explore 
    different aspects of the dataset.
    """)
    
    # Add a loading spinner while data is being loaded
    with st.spinner('Loading data... This may take a moment for large datasets.'):
        # Load the data
        df = load_data()
    
    if df is not None:
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Show basic dataset info
        st.sidebar.info(f"Dataset contains {len(df):,} records")
        
        # Create a copy of the filtered data to avoid modifying the cached version
        filtered_df = df.copy()
        
        # Year range slider
        if 'publish_year' in filtered_df.columns:
            # Get valid years (non-null)
            valid_years = filtered_df['publish_year'].dropna()
            if not valid_years.empty:
                min_year = int(valid_years.min())
                max_year = int(valid_years.max())
                
                # Set default range to show most recent 5 years if dataset is large
                default_min = max(min_year, max_year - 5) if (max_year - min_year) > 5 else min_year
                
                year_range = st.sidebar.slider(
                    "Select Year Range",
                    min_value=min_year,
                    max_value=max_year,
                    value=(default_min, max_year),
                    help="Filter papers by publication year"
                )
                
                # Filter data based on year range
                filtered_df = filtered_df[
                    (filtered_df['publish_year'] >= year_range[0]) & 
                    (filtered_df['publish_year'] <= year_range[1])
                ]
        
        # Journal selection - only show if we have a reasonable number of journals
        if 'journal' in filtered_df.columns:
            # Get top N journals to avoid overwhelming the selectbox
            top_journals = filtered_df['journal'].value_counts().head(50).index.tolist()
            journals = ['All'] + sorted(top_journals)
            
            selected_journal = st.sidebar.selectbox(
                "Select Journal (Top 50 shown)",
                journals,
                index=0,
                help="Select a journal to filter by (only top 50 by paper count shown)"
            )
            
            if selected_journal != 'All':
                filtered_df = filtered_df[filtered_df['journal'] == selected_journal]
        
        # Display dataset info with filtered counts
        st.sidebar.markdown("---")
        st.sidebar.info(f"**Filtered Dataset Info:**\n"
                      f"- Papers: {len(filtered_df):,} of {len(df):,}\n"
                      f"- Time period: {year_range[0] if 'year_range' in locals() else 'N/A'} - {year_range[1] if 'year_range' in locals() else 'N/A'}")
        
        # Add a progress bar for visual feedback
        progress_bar = st.progress(0)
        
        # Main content
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Show visualizations only if we have data
            if not filtered_df.empty:
                # Update progress
                progress_bar.progress(20)
                
                # Plot publications over time
                plot_publications_over_time(filtered_df)
                progress_bar.progress(40)
                
                # Show top journals only if we have journal data
                if 'journal' in filtered_df.columns and (selected_journal == 'All' or len(filtered_df) > 0):
                    plot_top_journals(filtered_df)
                progress_bar.progress(60)
                
                # Generate word cloud if we have titles
                if 'title' in filtered_df.columns and not filtered_df['title'].empty:
                    generate_wordcloud(filtered_df)
                progress_bar.progress(80)
            else:
                st.warning("No data available for the selected filters. Try adjusting your filter criteria.")
        
        with col2:
            # Show a sample of the filtered data
            st.subheader("Dataset Sample")
            if not filtered_df.empty:
                # Display a sample of the data
                sample_cols = [col for col in ['title', 'journal', 'publish_year'] if col in filtered_df.columns]
                st.dataframe(filtered_df[sample_cols].head(10))
                
                # Download button for filtered data with more options
                st.markdown("### Download Options")
                
                # Let user select which columns to include
                all_columns = filtered_df.columns.tolist()
                default_cols = ['title', 'journal', 'publish_year', 'title_word_count']
                selected_cols = st.multiselect(
                    "Select columns to include in download:",
                    options=all_columns,
                    default=default_cols,
                    help="Select which columns to include in the downloaded file"
                )
                
                # File format selection
                file_format = st.radio(
                    "Select file format:",
                    ["CSV", "Excel"],
                    index=0,
                    horizontal=True
                )
                
                # Prepare data for download
                download_data = filtered_df[selected_cols] if selected_cols else filtered_df
                
                # Create download button based on selected format
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if file_format == "CSV":
                    try:
                        csv = download_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 Download as CSV",
                            data=csv,
                            file_name=f"cord19_filtered_{timestamp}.csv",
                            mime='text/csv',
                            help="Download the filtered dataset as a CSV file"
                        )
                    except Exception as e:
                        st.error(f"Error generating CSV: {str(e)}")
                        st.warning("Please try again or select a different format.")
                else:  # Excel
                    try:
                        # Check if xlsxwriter is installed
                        import xlsxwriter
                        import io
                        
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            download_data.to_excel(writer, index=False, sheet_name='CORD19_Data')
                            
                        excel_data = buffer.getvalue()
                        st.download_button(
                            label="💾 Download as Excel",
                            data=excel_data,
                            file_name=f"cord19_filtered_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="Download the filtered dataset as an Excel file"
                        )
                    except ImportError:
                        st.error("Excel export requires the 'xlsxwriter' package.")
                        st.warning("Please install it by running: pip install xlsxwriter")
                        st.info("Alternatively, you can download the data in CSV format.")
                    except Exception as e:
                        st.error(f"Error generating Excel file: {str(e)}")
                        st.warning("Please try again or select CSV format instead.")
                
                # Calculate and show download size estimate
                try:
                    size_bytes = download_data.memory_usage(deep=True).sum()
                    size_mb = size_bytes / (1024 * 1024)
                    st.caption(f"Estimated download size: {size_mb:.2f} MB")
                except Exception as e:
                    st.caption("Download size estimation not available")
                
                # Show data summary
                with st.expander("📊 Data Summary", expanded=False):
                    st.json({
                        "Total Papers": len(filtered_df),
                        "Unique Journals": filtered_df['journal'].nunique() if 'journal' in filtered_df.columns else 0,
                        "Earliest Publication Year": int(filtered_df['publish_year'].min()) if 'publish_year' in filtered_df.columns and not filtered_df['publish_year'].isna().all() else None,
                        "Latest Publication Year": int(filtered_df['publish_year'].max()) if 'publish_year' in filtered_df.columns and not filtered_df['publish_year'].isna().all() else None,
                        "Average Title Length (words)": round(filtered_df['title'].str.split().str.len().mean(), 1) if 'title' in filtered_df.columns and not filtered_df['title'].isna().all() else 0
                    })
            
            # Add a clear filters button
            if st.sidebar.button("🔄 Clear All Filters"):
                st.experimental_rerun()
        
        # Complete the progress bar
        progress_bar.progress(100)
        progress_bar.empty()

if __name__ == "__main__":
    main()
