# CORD-19 Research Explorer

This project provides an interactive dashboard for exploring the CORD-19 dataset, which contains metadata about COVID-19 research papers. The application allows users to filter and visualize publication trends, top journals, and common words in paper titles.

## Features

- Interactive visualizations of publication trends over time
- Top publishing journals analysis
- Word cloud of common terms in paper titles
- Filter data by year range and journal
- Download filtered data as CSV

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/secbyteX03/Frameworks_Assignment.git
   cd Frameworks_Assignment
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your `metadata.csv` file in the project directory.

## Usage

1. First, run the data cleaning script:
   ```bash
   python cord19_analysis.py
   ```
   This will create a `cleaned_metadata.csv` file.

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your web browser to the URL shown in the terminal (usually http://localhost:8501).

## Project Structure

- `cord19_analysis.py`: Script for loading, exploring, and cleaning the CORD-19 metadata
- `app.py`: Streamlit application for interactive data exploration
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Data Source

The CORD-19 dataset is available on Kaggle:
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

## License

This project is for educational purposes as part of the PLP Python course week 8 Assignment.
