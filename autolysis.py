# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "pandas",
#   "matplotlib",
#   "httpx",
#   "chardet",
#   "ipykernel",
#   "openai",
#   "numpy",
#   "python-dotenv",
#   "scipy",
#   "scikit-learn"
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from dotenv import load_dotenv
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_env_variables():
    """Load environment variables and validate configurations."""
    load_dotenv()
    api_key = os.getenv("AIPROXY_TOKEN")
    if not api_key:
        print("Error: AIPROXY_TOKEN is not set. Ensure the token is loaded in the environment.")
        sys.exit(1)
    return api_key

def configure_openai(api_key):
    """Set OpenAI API configurations."""
    openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
    openai.api_key = api_key

def get_llm_response(prompt, model="gpt-4o-mini", max_tokens=800):
    """Calls the OpenAI API with the given prompt and retrieves the response."""
    try:
        print("Sending request to LLM...")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        print("LLM Raw Response:", response)
        if "choices" in response and len(response.choices) > 0:
            return response.choices[0].message["content"].strip()
        else:
            return "No analysis provided by LLM."
    except Exception as e:
        print(f"Error with LLM: {e}")
        return "LLM analysis failed."

def load_dataset(file_path):
    """Load a dataset from a CSV file."""
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        sys.exit(1)
    try:
        return pd.read_csv(file_path, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def calculate_advanced_statistics(data):
    """Calculate advanced statistics like skewness and kurtosis for numerical columns."""
    stats = {}
    for column in data.select_dtypes(include=[np.number]).columns:
        stats[column] = {
            "Skewness": skew(data[column].dropna()),
            "Kurtosis": kurtosis(data[column].dropna())
        }
    return stats

def perform_pca(data):
    """Perform PCA on scaled numerical data."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.select_dtypes(include=[np.number]).dropna(axis=1))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    return pca.explained_variance_ratio_

def generate_visualizations(data, output_dir):
    """Generate and save visualizations for the dataset."""
    correlation_matrix = data.corr(numeric_only=True)
    os.makedirs(output_dir, exist_ok=True)

    # Correlation Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()

    # Distribution Plots
    for column in data.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column].dropna(), kde=True, bins=30, color="blue")
        plt.axvline(data[column].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean')
        plt.axvline(data[column].median(), color='green', linestyle='dashed', linewidth=1, label='Median')
        plt.legend()
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"{column}_distribution.png"))
        plt.close()

def prepare_llm_prompt(data, missing_values, outlier_info, advanced_stats, pca_ratios):
    """Prepare a structured prompt for the LLM."""
    sample_data = data.head(3).to_dict(orient="records")
    return f"""
You are an AI data analyst. Here's the dataset summary:
- Number of Rows: {data.shape[0]}
- Number of Columns: {data.shape[1]}
- Columns: {list(data.columns)}
- Data Types: {data.dtypes.to_dict()}
- Missing Values: {missing_values.to_dict()}
- Outliers Detected: {outlier_info}
- Advanced Stats (Skewness & Kurtosis): {advanced_stats}
- PCA Result: Explained Variance Ratios: {pca_ratios}
- Sample Data: {sample_data}

Your task:
1. Analyze the dataset.
2. Highlight important trends, outliers, and correlations.
3. Suggest potential applications or interpretations of the data.
4. Provide actionable insights, implications, and recommendations.

Be concise, structured, and professional.
"""

def generate_readme_content(data, missing_values, advanced_stats, llm_analysis, output_dir):
    """Generate content for the README file."""
    content = f"""
# Automated Analysis Report

## Dataset Overview
- **Rows**: {data.shape[0]}
- **Columns**: {data.shape[1]}
- **Missing Values**:
{missing_values.to_string()}

## Key Insights
{llm_analysis}

## Advanced Stats
| Column | Skewness | Kurtosis |
|--------|----------|----------|
"""
    for column, stats in advanced_stats.items():
        content += f"| {column} | {stats['Skewness']:.2f} | {stats['Kurtosis']:.2f} |\n"

    content += """

## Visualizations
### Correlation Matrix
![Correlation Matrix](correlation_matrix.png)

### Distributions
"""
    for column in data.select_dtypes(include=[np.number]).columns:
        content += f"![{column} Distribution]({column}_distribution.png)\n"

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(content)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    api_key = load_env_variables()
    configure_openai(api_key)

    data = load_dataset(csv_file)
    summary_stats = data.describe(include="all").transpose()
    missing_values = data.isnull().sum()
    advanced_stats = calculate_advanced_statistics(data)
    pca_ratios = perform_pca(data)

    # Example: Detect outliers using z-scores
    outlier_info = {}
    for column in data.select_dtypes(include=[np.number]).columns:
        z_scores = (data[column] - data[column].mean()) / data[column].std()
        outlier_info[column] = len(data[np.abs(z_scores) > 3][column])

    output_dir = os.path.splitext(os.path.basename(csv_file))[0]
    generate_visualizations(data, output_dir)

    llm_prompt = prepare_llm_prompt(data, missing_values, outlier_info, advanced_stats, pca_ratios)
    llm_analysis = get_llm_response(llm_prompt)

    generate_readme_content(data, missing_values, advanced_stats, llm_analysis, output_dir)

    print(f"Analysis complete. Outputs saved in {output_dir}/README.md and PNG files.")
