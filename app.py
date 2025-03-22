import pandas as pd
import ollama
import csv
import re
import os
import concurrent.futures

# File paths
input_file = "clean_vulnerability_data_v2.csv"
output_file = "vulnerabilities_fixed.csv"
progress_file = "progress.log"
num_workers = 2  # Adjust based on your CPU cores (e.g., set to os.cpu_count())

# Function to extract only code from AI response
def extract_code(response_text):
    """
    Extracts only the code block from the AI response.
    Removes explanations, keeping only the fixed code.
    """
    match = re.search(r"```(?:[a-zA-Z]+\n)?(.*?)```", response_text, re.DOTALL)
    return match.group(1).strip() if match else response_text.strip()

# Function to get last processed index from progress file
def get_last_processed_index():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            last_line = f.readline().strip()
            return int(last_line) if last_line.isdigit() else -1
    return -1

# Function to update progress file
def update_progress(index):
    with open(progress_file, "w") as f:
        f.write(str(index))

# Check if output file exists and contains data
def file_has_data(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

# Read input CSV
df = pd.read_csv(input_file)

# Ensure headers exist in output file
if not file_has_data(output_file):
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vulnerability_type", "vulnerable_code", "fixed_code"])

# Get last processed index
last_processed_index = get_last_processed_index()

# Filter unprocessed rows
unprocessed_rows = df.iloc[last_processed_index + 1:]  # Skip already processed rows

# Function to process a single row
def process_row(index, row):
    """
    Processes a single row, generates the fixed code using AI, and appends it to the output CSV.
    """
    print(f"Processing row {index+1}/{len(df)} - {row['vulnerability_type']} vulnerability")

    # Prepare prompt
    prompt = f"""
    You are an expert in secure coding. Below is a {row['vulnerability_type']} vulnerability with an incomplete fix.

    Vulnerable Code:
    {row['vulnerable_code']}

    Incomplete Fixed Code:
    {row['fixed_code']}

    Complete only the missing parts while keeping the correct existing code unchanged.
    Do not include explanations, only return the corrected code inside triple backticks (```).
    """

    try:
        # Get AI-generated completion
        response = ollama.chat(model="wizardcoder", messages=[{"role": "user", "content": prompt}])
        raw_fixed_code = response['message']['content']

        # Extract only the code
        fixed_code = extract_code(raw_fixed_code)

        # Append processed row to CSV (thread-safe)
        with open(output_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row['vulnerability_type'], row['vulnerable_code'], fixed_code])

        # Update progress
        update_progress(index)

    except Exception as e:
        print(f"Error processing row {index+1}: {e}")

# Process rows in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(lambda row: process_row(row[0], row[1]), unprocessed_rows.iterrows())

print(f"Processing complete. Updated file saved as {output_file}")