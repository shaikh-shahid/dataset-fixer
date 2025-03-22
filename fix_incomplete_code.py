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
fixed_only_file = "vulnerabilities_fixed_only.csv"
progress_file = "completion_progress.log"
error_log_file = "error_log.txt"

# Configuration
MAX_WORKERS = 4
MEMORY_THRESHOLD = 75
BATCH_SIZE = 20
MIN_CODE_LENGTH = 100  # Minimum acceptable length for completed code
MAX_RETRIES = 5  # Increased retries

def get_system_memory_usage():
    """Get current system memory usage percentage"""
    return psutil.virtual_memory().percent

def is_system_overloaded():
    """Check if system resources are overloaded"""
    return get_system_memory_usage() > MEMORY_THRESHOLD

def log_error(message):
    """Log errors to file"""
    with open(error_log_file, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def is_incomplete_code(code):
    """Check if the code is incomplete by looking for common markers and structure"""
    markers = [
        "To fix the",
        "Here is",
        "Here's",
        "Below is",
        "The following"
    ]
    
    # Check for markers
    has_markers = any(marker.lower() in str(code).lower() for marker in markers)
    
    # Check if code looks incomplete (missing closing braces, etc.)
    code_str = str(code)
    opening_braces = code_str.count("{")
    closing_braces = code_str.count("}")
    
    # Code is incomplete if:
    # 1. Contains explanation markers, or
    # 2. Has mismatched braces, or
    # 3. Ends with explanation text
    return (has_markers or 
            opening_braces != closing_braces or 
            any(code_str.strip().endswith(marker) for marker in markers))

def validate_completed_code(original_code, completed_code, vulnerability_type):
    """Validate that the completed code is acceptable"""
    if not completed_code or len(completed_code) < MIN_CODE_LENGTH:
        return False
        
    # Check if it's just a small fragment
    if len(completed_code) < len(original_code) * 0.5:
        return False
    
    # Check if it maintains the core functionality
    original_lines = set(line.strip() for line in original_code.split('\n') if line.strip())
    completed_lines = set(line.strip() for line in completed_code.split('\n') if line.strip())
    
    # At least 50% of original lines should be present in completed code
    common_lines = original_lines.intersection(completed_lines)
    if len(common_lines) < len(original_lines) * 0.5:
        return False
    
    return True

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
        time.sleep(5)
    
    # Extract the language from the code block if possible
    lang_match = re.search(r"```(\w+)", incomplete_code)
    language = lang_match.group(1) if lang_match else "unknown"
    
    # Clean up the incomplete code first
    cleaned_incomplete = clean_code(incomplete_code)
    
    prompt = f"""As a secure coding expert, complete this {vulnerability_type} vulnerability fix in {language}.
The code below is incomplete. Your task is to provide a COMPLETE, working solution that includes ALL necessary parts:

{cleaned_incomplete}

Requirements:
1. Return the COMPLETE code, not just the missing parts
2. Include ALL security measures and validation
3. Keep all existing security features
4. Include ALL necessary imports and class definitions
5. Ensure proper error handling
6. Return only code, no explanations
7. Code must be complete and runnable

The code should handle the {vulnerability_type} vulnerability completely and securely."""
    
    retry_count = 0
    best_completion = None
    max_length = 0
    
    while retry_count < MAX_RETRIES:
        try:
            response = ollama.chat(
                model="codellama:7b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a secure coding expert. Always provide complete, secure, runnable code."
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            
            completed_code = response['message']['content']
            cleaned_completion = clean_code(completed_code)
            
            # Keep the best (longest) valid completion
            if validate_completed_code(cleaned_incomplete, cleaned_completion, vulnerability_type):
                if len(cleaned_completion) > max_length:
                    best_completion = cleaned_completion
                    max_length = len(cleaned_completion)
                    if max_length > MIN_CODE_LENGTH * 2:  # If we got a good completion, stop early
                        break
            
            retry_count += 1
            if retry_count < MAX_RETRIES:
                time.sleep(2)
        except Exception as e:
            log_error(f"Error in completion attempt {retry_count}: {str(e)}")
            retry_count += 1
            time.sleep(2)
    
    return best_completion

def process_row(args):
    """Process a single row with error handling"""
    index, row = args
    try:
        if is_incomplete_code(str(row['fixed_code'])):
            original_code = str(row['fixed_code'])
            completed_code = complete_code_with_ollama(original_code, row['vulnerability_type'])
            
            if completed_code and validate_completed_code(original_code, completed_code, row['vulnerability_type']):
                return index, completed_code, True
            else:
                log_error(f"Failed to generate valid completion for row {index}")
                return index, original_code, False
    except Exception as e:
        log_error(f"Error processing row {index}: {str(e)}")
        return index, str(row['fixed_code']), False
    return index, str(row['fixed_code']), False

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
                    index, cleaned_code, is_valid = future.result()
                    if is_valid:
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