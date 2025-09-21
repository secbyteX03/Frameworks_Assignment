import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import streamlit as st
from datetime import datetime

# Set style for plots
sns.set(style="whitegrid")

def load_data():
    """Load the CORD-19 metadata CSV file."""
    try:
        df = pd.read_csv('metadata.csv', low_memory=False)
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading the file: {e}")
        return None

def explore_data(df):
    """Perform initial data exploration."""
    print("\n=== Data Exploration ===")
    
    # Basic information
    print("\n1. DataFrame Shape:", df.shape)
    
    # Data types and non-null counts
    print("\n2. Data Types and Non-null Counts:")
    print(df.info())
    
    # Check for missing values
    print("\n3. Missing Values:")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0])
    
    # Basic statistics for numerical columns
    print("\n4. Basic Statistics for Numerical Columns:")
    print(df.describe())
    
    return df

def clean_data(df):
    """Clean and prepare the data for analysis."""
    print("\n=== Data Cleaning ===")
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Convert publish_time to datetime and extract year
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    df_clean['publish_year'] = df_clean['publish_time'].dt.year
    
    # Handle missing values in key columns
    # For this analysis, we'll drop rows where title is missing
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['title', 'publish_year'])
    print(f"Dropped {initial_count - len(df_clean)} rows with missing titles or years")
    
    # Create a column for title word count
    df_clean['title_word_count'] = df_clean['title'].apply(lambda x: len(str(x).split()))
    
    return df_clean

if __name__ == "__main__":
    # Load the data
    print("Loading CORD-19 metadata...")
    df = load_data()
    
    if df is not None:
        # Perform initial exploration
        explore_data(df)
        
        # Clean the data
        df_clean = clean_data(df)
        
        # Save the cleaned data for later use
        df_clean.to_csv('cleaned_metadata.csv', index=False)
        print("\nCleaned data saved to 'cleaned_metadata.csv'")
