import pandas as pd
import ollama
import re
import os
import psutil
import concurrent.futures
from tqdm import tqdm
import time
import gc

# File paths
input_file = "vulnerabilities_fixed.csv"
output_file = "vulnerabilities_fixed_completed.csv"
progress_file = "completion_progress.log"

# Configuration
MAX_WORKERS = 4  # Increased for EC2
MEMORY_THRESHOLD = 75  # Lower threshold for EC2
BATCH_SIZE = 20  # Increased batch size for EC2
SAVE_INTERVAL = 50  # Save progress every 50 rows

def get_system_memory_usage():
    """Get current system memory usage percentage"""
    return psutil.virtual_memory().percent

def is_system_overloaded():
    """Check if system resources are overloaded"""
    return get_system_memory_usage() > MEMORY_THRESHOLD

def is_incomplete_code(code):
    """Check if the code is incomplete by looking for common markers"""
    markers = [
        "To fix the",
        "Here is",
        "Here's",
        "Below is",
        "The following"
    ]
    return any(marker.lower() in code.lower() for marker in markers)

def extract_code_blocks(text):
    """Extract code blocks between triple backticks"""
    code_blocks = re.findall(r"```(?:java|python|javascript|cpp|c\+\+|c#|csharp|ruby|php|go|rust|typescript|js)?\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)
    return text

def clean_code(text):
    """Clean the code by removing explanations and keeping only code blocks"""
    # First try to extract code blocks
    code = extract_code_blocks(text)
    
    # Remove any remaining markdown or explanatory text
    lines = code.split('\n')
    code_lines = []
    
    for line in lines:
        # Skip explanatory text or markdown
        if line.strip().startswith(('To fix', 'Here is', 'Here\'s', 'The following')):
            continue
        code_lines.append(line)
    
    return '\n'.join(code_lines).strip()

def get_last_processed_index():
    """Get the last processed index from progress file"""
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            last_line = f.readline().strip()
            return int(last_line) if last_line.isdigit() else -1
    return -1

def update_progress(index):
    """Update progress file"""
    with open(progress_file, "w") as f:
        f.write(str(index))

def complete_code_with_ollama(incomplete_code, vulnerability_type):
    """Use Ollama to complete the code"""
    if is_system_overloaded():
        time.sleep(5)  # Wait for system resources to free up
        
    # Extract the language from the code block if possible
    lang_match = re.search(r"```(\w+)", incomplete_code)
    language = lang_match.group(1) if lang_match else "unknown"
        
    prompt = f"""You are a secure coding expert. Focus on completing this {vulnerability_type} vulnerability fix in {language}.
The code below is incomplete. Complete it while preserving the existing secure parts:

{incomplete_code}

Important:
1. Keep all existing security measures
2. Complete any missing validation or security checks
3. Return ONLY the code, no explanations
4. Maintain the same programming language and style
5. Include all necessary imports and dependencies"""
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = ollama.chat(
                model="codellama:7b",
                messages=[
                    {"role": "system", "content": "You are a secure coding expert. Respond only with complete, secure code."},
                    {"role": "user", "content": prompt}
                ]
            )
            completed_code = response['message']['content']
            
            # Verify the response contains actual code
            if "```" in completed_code and len(completed_code) > 50:
                return completed_code
            else:
                print(f"Retry {retry_count + 1}: Response too short or no code block found")
                retry_count += 1
                time.sleep(2)
        except Exception as e:
            print(f"Error on attempt {retry_count + 1}: {e}")
            retry_count += 1
            time.sleep(2)
    
    return None

def process_row(args):
    """Process a single row with error handling"""
    index, row = args
    try:
        if is_incomplete_code(str(row['fixed_code'])):
            completed_code = complete_code_with_ollama(row['fixed_code'], row['vulnerability_type'])
            if completed_code:
                cleaned_code = clean_code(completed_code)
                return index, cleaned_code
    except Exception as e:
        print(f"Error processing row {index}: {e}")
    return index, None

def main():
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get last processed index
    last_processed_index = get_last_processed_index()
    
    # Create a copy of the dataframe for processing
    processed_df = df.copy()
    
    # Get unprocessed rows
    unprocessed_rows = list(df.iloc[last_processed_index + 1:].iterrows())
    total_rows = len(unprocessed_rows)
    
    print(f"Processing {total_rows} rows with {MAX_WORKERS} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Process rows in batches
        for batch_start in range(0, total_rows, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_rows)
            batch = unprocessed_rows[batch_start:batch_end]
            
            # Process batch
            futures = [executor.submit(process_row, row) for row in batch]
            
            # Handle results
            for future in concurrent.futures.as_completed(futures):
                try:
                    index, cleaned_code = future.result()
                    if cleaned_code:
                        processed_df.at[index, 'fixed_code'] = cleaned_code
                        processed_df.to_csv(output_file, index=False)
                        update_progress(index)
                except Exception as e:
                    print(f"Error processing batch: {e}")
            
            # Force garbage collection between batches
            gc.collect()
            
            # Check system resources
            if is_system_overloaded():
                print("System resources high, pausing for 10 seconds...")
                time.sleep(10)
    
    print(f"Processing complete. Updated file saved as {output_file}")

if __name__ == "__main__":
    main() 