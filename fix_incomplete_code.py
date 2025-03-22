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
debug_file = "debug_log.txt"

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

def debug_log(message):
    """Log debug information"""
    print(message)  # Print to console
    with open(debug_file, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def is_incomplete_code(code):
    """Check if the code is incomplete Java code"""
    code_str = str(code).strip()
    
    # Check if the text starts with an explanation (case insensitive)
    starts_with_explanation = bool(re.match(r'^\s*to\s+fix\s+', code_str.lower()))
    
    # Look for Java code blocks
    code_blocks = re.findall(r'```java\s*(.*?)```', code_str, re.DOTALL)
    
    # If it starts with "To fix" and has no complete code block, it's incomplete
    if starts_with_explanation:
        if not code_blocks:
            debug_log(f"Found 'To fix' but no complete Java code block")
            return True
        
        # Check if the last code block is incomplete
        last_block = code_blocks[-1].strip()
        has_missing_braces = last_block.count('{') != last_block.count('}')
        has_missing_parens = last_block.count('(') != last_block.count(')')
        ends_with_control = last_block.rstrip().endswith(('catch', 'try', 'else', 'finally', '{', '}', ';'))
        
        debug_log(f"Java code analysis - Missing braces: {has_missing_braces}, "
                 f"Missing parentheses: {has_missing_parens}, "
                 f"Ends abruptly: {ends_with_control}")
        
        return has_missing_braces or has_missing_parens or ends_with_control
    
    return False  # If it doesn't start with "To fix", assume it's complete

def validate_completed_code(original_code, completed_code, vulnerability_type):
    """Validate that the completed Java code is acceptable"""
    if not completed_code or len(completed_code) < MIN_CODE_LENGTH:
        debug_log(f"Validation failed: Code too short or empty. Length: {len(completed_code) if completed_code else 0}")
        return False
    
    # Clean both codes for comparison
    original_clean = clean_code(original_code)
    completed_clean = clean_code(completed_code)
    
    # Java-specific validation
    if not (completed_clean.count('{') == completed_clean.count('}') and
            completed_clean.count('(') == completed_clean.count(')')):
        debug_log(f"Validation failed: Mismatched braces or parentheses in Java code")
        return False
    
    # Check for essential Java elements
    has_proper_end = completed_clean.strip().endswith('}') or completed_clean.strip().endswith(';')
    has_statements = completed_clean.count(';') > 0
    
    if not (has_proper_end and has_statements):
        debug_log(f"Validation failed: Missing essential Java elements")
        return False
    
    # Check if it maintains the core functionality
    original_lines = set(line.strip() for line in original_clean.split('\n') if line.strip())
    completed_lines = set(line.strip() for line in completed_clean.split('\n') if line.strip())
    
    # At least 50% of original lines should be present in completed code
    common_lines = original_lines.intersection(completed_lines)
    if len(common_lines) < len(original_lines) * 0.5:
        debug_log(f"Validation failed: Too few common lines with original code")
        return False
    
    debug_log(f"Java code validation passed. Length: {len(completed_clean)}, Common lines: {len(common_lines)}")
    return True

def extract_code_blocks(text):
    """Extract code blocks between triple backticks"""
    code_blocks = re.findall(r"```(?:java|python|javascript|cpp|c\+\+|c#|csharp|ruby|php|go|rust|typescript|js)?\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks)
    return text

def clean_code(text):
    """Clean the code by extracting Java code blocks"""
    # Look for Java code blocks
    code_blocks = re.findall(r'```java\s*(.*?)```', text, re.DOTALL)
    
    if code_blocks:
        # Get the last code block (the fixed version)
        code = code_blocks[-1].strip()
        return code
    
    # If no code blocks found, return the cleaned text
    return text.strip()

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
    
    # Clean up the incomplete code first
    cleaned_incomplete = clean_code(incomplete_code)
    
    # Extract the explanation part
    explanation_match = re.match(r'^\s*to\s+fix\s+(.*?)```java', incomplete_code, re.DOTALL | re.IGNORECASE)
    explanation = explanation_match.group(1).strip() if explanation_match else ""
    
    prompt = f"""Complete this {vulnerability_type} vulnerability fix. 
The explanation of the fix is: {explanation}

The incomplete code is:
{cleaned_incomplete}

Requirements:
1. Complete the code by adding any missing parts
2. Ensure all security measures are maintained
3. Add proper error handling
4. Make sure all blocks are properly closed
5. Return ONLY the complete code without any explanations
6. Include all necessary imports

Return the complete code inside triple backticks with the java language specifier."""
    
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
                        "content": "You are a secure coding expert. Complete the code without adding explanations."
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            
            completed_code = response['message']['content']
            cleaned_completion = clean_code(completed_code)
            
            # Validate the completion
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
    
    # Count incomplete rows first
    incomplete_count = 0
    for index, row in df.iterrows():
        if is_incomplete_code(str(row['fixed_code'])):
            incomplete_count += 1
            debug_log(f"Row {index} identified as incomplete. Vulnerability type: {row['vulnerability_type']}")
    
    debug_log(f"Found {incomplete_count} incomplete rows out of {len(df)} total rows")
    print(f"Found {incomplete_count} incomplete rows that need fixing...")
    
    # Get unprocessed rows
    unprocessed_rows = list(df.iloc[last_processed_index + 1:].iterrows())
    total_rows = len(unprocessed_rows)
    
    print(f"Processing {total_rows} remaining rows with {MAX_WORKERS} workers...")
    
    processed_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Process rows in batches
        for batch_start in range(0, total_rows, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_rows)
            batch = unprocessed_rows[batch_start:batch_end]
            
            debug_log(f"Processing batch {batch_start//BATCH_SIZE + 1}, rows {batch_start} to {batch_end}")
            
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
                        processed_count += 1
                        debug_log(f"Successfully processed row {index}. Total processed: {processed_count}")
                except Exception as e:
                    error_msg = f"Error processing batch: {str(e)}"
                    print(error_msg)
                    debug_log(error_msg)
            
            # Force garbage collection between batches
            gc.collect()
            
            # Check system resources
            if is_system_overloaded():
                debug_log("System resources high, pausing for 10 seconds...")
                time.sleep(10)
    
    debug_log(f"Processing complete. Processed {processed_count} rows out of {incomplete_count} incomplete rows")
    print(f"Processing complete. Processed {processed_count} rows out of {incomplete_count} incomplete rows")
    print(f"Updated file saved as {output_file}")
    print(f"Check {debug_file} for detailed processing log")

if __name__ == "__main__":
    main() 