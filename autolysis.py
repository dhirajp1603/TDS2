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

# Load environment variables
load_dotenv()

# Set OpenAI API configurations
API_BASE = "https://aiproxy.sanand.workers.dev/openai/v1"
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
MODEL = "gpt-4o-mini"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 800))  # Make max_tokens configurable

# Configure OpenAI API client
openai.api_base = API_BASE
openai.api_key = AIPROXY_TOKEN

# Validate OpenAI API Key
if not openai.api_key:
    print("Error: AIPROXY_TOKEN is not set. Ensure the token is loaded in the environment.")
    sys.exit(1)

# Function to call the LLM
def get_llm_response(prompt):
    """
    Calls the OpenAI API with the given prompt and retrieves the response.
    """
    try:
        print("Sending request to LLM...")
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an AI analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS
        )
        print("LLM Raw Response:", response)
        if "choices" in response and len(response.choices) > 0:
            return response.choices[0].message["content"].strip()
        else:
            return "No analysis provided by LLM."
    except Exception as e:
        print(f"Error with LLM: {e}")
        return "LLM analysis failed."

# Validate command-line arguments
if len(sys.argv) != 2:
    print("Usage: python autolysis.py <dataset.csv>")
    sys.exit(1)

# Load the dataset
csv_file = sys.argv[1]
if not os.path.exists(csv_file):
    print(f"File {csv_file} not found.")
    sys.exit(1)

try:
    data = pd.read_csv(csv_file, encoding='ISO-8859-1')
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Generate summary statistics
summary_stats = data.describe(include="all").transpose()
missing_values = data.isnull().sum()
correlation_matrix = data.corr(numeric_only=True)

# Advanced statistical analysis
additional_stats = {}
for column in data.select_dtypes(include=[np.number]).columns:
    additional_stats[column] = {
        "Skewness": skew(data[column].dropna()),
        "Kurtosis": kurtosis(data[column].dropna())
    }

# Dimensionality reduction using PCA
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.select_dtypes(include=[np.number]).dropna(axis=1))
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

# Define the output directory based on the dataset name
dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
output_dir = os.path.join(dataset_name)

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix")
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
plt.close()

# Generate distribution plots for numerical columns
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

# Perform deeper statistical analysis
# Example: Calculate z-scores to detect outliers
outlier_info = {}
for column in data.select_dtypes(include=[np.number]).columns:
    col_mean = data[column].mean()
    col_std = data[column].std()
    z_scores = (data[column] - col_mean) / col_std
    outliers = data[np.abs(z_scores) > 3][column]
    outlier_info[column] = len(outliers)

# Prepare data for LLM
sample_data = data.head(3).to_dict(orient="records")  # Limit to 3 rows for efficiency
llm_prompt = f"""
You are an AI data analyst. Here's the dataset summary:
- Number of Rows: {data.shape[0]}
- Number of Columns: {data.shape[1]}
- Columns: {list(data.columns)}
- Data Types: {data.dtypes.to_dict()}
- Missing Values: {missing_values.to_dict()}
- Outliers Detected: {outlier_info}
- Advanced Stats (Skewness & Kurtosis): {additional_stats}
- PCA Result: Explained Variance Ratios: {pca.explained_variance_ratio_}
- Sample Data: {sample_data}

Your task:
1. Analyze the dataset.
2. Highlight important trends, outliers, and correlations.
3. Suggest potential applications or interpretations of the data.
4. Provide actionable insights, implications, and recommendations.

Be concise, structured, and professional.
"""

llm_analysis = get_llm_response(llm_prompt)

# Generate README.md content
markdown_content = f"""
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

for column, stats in additional_stats.items():
    markdown_content += f"| {column} | {stats['Skewness']:.2f} | {stats['Kurtosis']:.2f} |\n"

markdown_content += """

## Visualizations
### Correlation Matrix
![Correlation Matrix](correlation_matrix.png)

### Distributions
"""

# Add distribution plots to the README content
for column in data.select_dtypes(include=[np.number]).columns:
    markdown_content += f"![{column} Distribution]({column}_distribution.png)\n"

# Save README.md inside the output directory
readme_path = os.path.join(output_dir, "README.md")
with open(readme_path, "w") as f:
    f.write(markdown_content)

print(f"Analysis complete. Outputs saved in {output_dir}/README.md and PNG files.")
