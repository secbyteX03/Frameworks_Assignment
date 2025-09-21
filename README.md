# CORD-19 Research Explorer

This project provides an interactive dashboard for exploring the CORD-19 dataset, which contains metadata about COVID-19 research papers. The application allows users to filter and visualize publication trends, top journals, and common words in paper titles with optimized performance for large datasets.

## 🌟 Features

- **Publication Trends**: Interactive visualization of research output over time
- **Top Journals Analysis**: Identify leading publishers with publication counts and percentages
- **Enhanced Word Cloud**: 
  - Memory-efficient processing of large datasets
  - Customizable color schemes
  - Adjustable word count
  - Optimized text processing with stopword removal
- **Responsive Design**: Works on various screen sizes
- **Interactive Filters**:
  - Filter by year range
  - Filter by journal
  - Adjust visualization parameters in real-time
- **Data Export**: Download filtered data as CSV

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/secbyteX03/Frameworks_Assignment.git
   cd Frameworks_Assignment
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
   The main dependencies include:
   - Streamlit
   - Pandas
   - Matplotlib
   - WordCloud
   - Seaborn
   - mplcursors (for interactive plots)

4. Place your `metadata.csv` file in the project directory or update the file path in the code if needed.

## 🖥️ Usage

1. **Data Preparation** (First time only):
   ```bash
   python cord19_analysis.py
   ```
   This will create a `cleaned_metadata.csv` file with processed data.

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   The application will start and automatically open in your default web browser at `http://localhost:8501`.

3. **Using the Dashboard**:
   - Use the sidebar filters to adjust the data view
   - Hover over charts for detailed information
   - Adjust word cloud settings using the interactive controls
   - Download filtered data using the export button

## 🛠️ Performance Optimizations

The application includes several optimizations for handling large datasets:
- Chunked data processing for memory efficiency
- Caching of expensive computations
- Optimized word cloud generation
- Progress indicators for long-running operations

## 📂 Project Structure

```
Frameworks_Assignment/
├── app.py                 # Main Streamlit application
├── cord19_analysis.py     # Data loading and preprocessing
├── requirements.txt       # Project dependencies
├── README.md              # This file
└── cleaned_metadata.csv   # Processed dataset (generated)
```

### Key Files:
- `app.py`: Contains the interactive dashboard with visualizations
- `cord19_analysis.py`: Handles data loading, cleaning, and preprocessing
- `requirements.txt`: Lists all Python package dependencies

## Data Source

The CORD-19 dataset is available on Kaggle:
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

## License

This project is for educational purposes as part of the PLP Python course week 8 Assignment.
