from flask import Flask, render_template, jsonify, send_file
import pandas as pd
from scholarly import scholarly
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import numpy as np
import json

app = Flask(__name__)

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Upload and Analyze Data
@app.route('/upload', methods=['POST'])
def upload():
    # Directly load the JSON data
    try:
        with open('output.json', 'r') as f:
            data = [json.loads(line) for line in f.readlines(5)] 

        # Convert JSON data to a DataFrame
        df = pd.DataFrame(data)

        # Perform analysis with a limit and threading for speed
        correlation, chart_data = analyze_data(df, max_researchers=40)

        return render_template('index.html', correlation=correlation)

    except Exception as e:
        print(f"Error during upload and analysis: {e}")
        return f"Error during upload and analysis: {e}"
    
# Download Report
@app.route('/download_report')
def download_report():
    return send_file('report.txt', as_attachment=True) 

# Generate Graph
@app.route('/generate_graph')
def generate_graph():
    try:
        # Directly load the JSON data
        with open('output.json', 'r') as f:
            data = [json.loads(line) for line in f.readlines()]

        # Convert JSON data to a DataFrame
        df = pd.DataFrame(data)

        # Perform citation scraping using threading and ensure max 40 unique researchers
        citation_data = []
        names_seen = set()
        names_for_citation = []

        for name in df['Name']:
            if name not in names_seen and len(names_for_citation) < 40:
                names_for_citation.append(name)
                names_seen.add(name)

        # Get citations using threading
        citations = get_citations_threaded(names_for_citation)
        citation_data = [{'Name': name, 'Citations': citations[name]} for name in names_for_citation if citations[name] > 0]
        
        # Create DataFrame with citation data
        citation_df = pd.DataFrame(citation_data)
        df = pd.merge(df, citation_df, on='Name', how='left')

        # Remove outliers using 95th percentile cap
        citation_upper = df['Citations'].quantile(0.95)
        amount_upper = df['Amount($)'].quantile(0.95)
        df = df[(df['Citations'] <= citation_upper) & (df['Amount($)'] <= amount_upper)]

        # Apply logarithmic scale for better visualization
        df['Log_Citations'] = df['Citations'].apply(lambda x: max(x, 1))
        df['Log_Citations'] = np.log10(df['Log_Citations'])
        df['Log_Amount'] = df['Amount($)'].apply(lambda x: max(x, 1))
        df['Log_Amount'] = np.log10(df['Log_Amount'])

        # Generate scatter plot with regression line
        plt.figure(figsize=(8, 6))
        sns.regplot(x='Log_Citations', y='Log_Amount', data=df, scatter_kws={'s': 80}, line_kws={'color': 'red'})
        plt.title('Citations vs. Grant Amounts (Log Scale with Correlation Line)')
        plt.xlabel('Log(Citations)')
        plt.ylabel('Log(Grant Amount ($))')

        # Save plot to a file
        plot_path = './static/graph.png'
        plt.savefig(plot_path)
        plt.close() 

        return render_template('index.html', correlation="Graph generated", graph_path='/static/graph.png')

    except Exception as e:
        print(f"Error generating graph: {e}")
        return f"Error generating graph: {e}"

# Analyze Data with a Limit and Threading
def analyze_data(df, max_researchers=40):
    try:
        df.columns = df.columns.str.strip()

        print("Detected Columns:", df.columns.tolist())

        # Ensure required columns exist
        required_columns = ['Name', 'Amount($)']
        for col in required_columns:
            if col not in df.columns:
                return f"Error: Column '{col}' not found. Available columns: {df.columns.tolist()}", []

        # Limit sample size to ensure fast analysis and avoid API rate limits
        df = df.head(max_researchers)

        # Perform citation scraping using threading
        citations = get_citations_threaded(list(set(df['Name'].tolist())))
        df['Citations'] = df['Name'].map(citations)

        # Generate correlation
        correlation = df['Citations'].corr(df['Amount($)'])
        return correlation, []
    except Exception as e:
        print(f"Error during analysis: {e}")
        return f"Error: {e}", []

# Scrape Google Scholar for Citations Using Threading
def get_citations_threaded(names):
    citations = {}
    
    def fetch_citation(name):
        try:
            search_query = scholarly.search_author(name)
            author = next(search_query, None)

            if author:
                scholarly.fill(author)
                citations[name] = author.get('citedby', 0)
                print(f"{name}: {citations[name]} citations")
            else:
                citations[name] = 0
        except Exception as e:
            print(f"Error fetching data for {name}: {e}")
            citations[name] = 0

    # Use ThreadPoolExecutor for faster data fetching
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(fetch_citation, names)
    
    return citations

if __name__ == '__main__':
    app.run(debug=True)
