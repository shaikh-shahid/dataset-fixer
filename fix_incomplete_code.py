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
    
    # Look for Java code blocks with more flexible pattern
    code_blocks = re.findall(r'```(?:java)?\s*(.*?)```', code_str, re.DOTALL) or \
                 re.findall(r'```\s*(.*?)```', code_str, re.DOTALL)
    
    debug_log(f"Code analysis - Starts with 'To fix': {starts_with_explanation}, "
             f"Number of code blocks found: {len(code_blocks)}")
    
    # If it starts with "To fix" and has no complete code block, it's incomplete
    if starts_with_explanation:
        if not code_blocks:
            debug_log(f"Found 'To fix' but no complete code block")
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
    # Look for Java code blocks with more flexible pattern
    code_blocks = re.findall(r'```(?:java)?\s*(.*?)```', text, re.DOTALL) or \
                 re.findall(r'```\s*(.*?)```', text, re.DOTALL)
    
    if code_blocks:
        # Get the last code block (the fixed version)
        code = code_blocks[-1].strip()
        debug_log(f"Found code block with length: {len(code)}")
        return code
    
    # If no code blocks found, return the cleaned text
    debug_log("No code blocks found, returning cleaned text")
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
    
    try:
        # Clean up the incomplete code first
        cleaned_incomplete = clean_code(incomplete_code)
        debug_log(f"Cleaned incomplete code length: {len(cleaned_incomplete)}")
        
        # Extract the explanation part
        explanation_match = re.match(r'^\s*to\s+fix\s+(.*?)```', incomplete_code, re.DOTALL | re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        debug_log(f"Extracted explanation length: {len(explanation)}")
        
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

Return the complete code inside triple backticks."""
        
        retry_count = 0
        best_completion = None
        max_length = 0
        
        while retry_count < MAX_RETRIES:
            try:
                debug_log(f"Attempt {retry_count + 1} to complete code")
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
                debug_log(f"Got completion of length: {len(cleaned_completion)}")
                
                # Validate the completion
                if validate_completed_code(cleaned_incomplete, cleaned_completion, vulnerability_type):
                    if len(cleaned_completion) > max_length:
                        best_completion = cleaned_completion
                        max_length = len(cleaned_completion)
                        debug_log(f"Found better completion of length: {max_length}")
                        if max_length > MIN_CODE_LENGTH * 2:  # If we got a good completion, stop early
                            break
                
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    time.sleep(2)
            except Exception as e:
                error_msg = f"Error in completion attempt {retry_count}: {str(e)}"
                debug_log(error_msg)
                log_error(error_msg)
                retry_count += 1
                time.sleep(2)
        
        return best_completion
    except Exception as e:
        error_msg = f"Error in complete_code_with_ollama: {str(e)}"
        debug_log(error_msg)
        log_error(error_msg)
        return None

def process_row(args):
    """Process a single row with error handling"""
    index, row = args
    try:
        debug_log(f"Starting to process row {index}")
        if is_incomplete_code(str(row['fixed_code'])):
            debug_log(f"Row {index} identified as incomplete, attempting to complete")
            original_code = str(row['fixed_code'])
            completed_code = complete_code_with_ollama(original_code, row['vulnerability_type'])
            
            if completed_code:
                debug_log(f"Got completion for row {index}, validating")
                if validate_completed_code(original_code, completed_code, row['vulnerability_type']):
                    debug_log(f"Validation passed for row {index}")
                    return index, completed_code, True
                else:
                    debug_log(f"Validation failed for row {index}")
            else:
                debug_log(f"No completion generated for row {index}")
            
            log_error(f"Failed to generate valid completion for row {index}")
            return index, original_code, False
        else:
            debug_log(f"Row {index} is already complete")
            return index, str(row['fixed_code']), False
    except Exception as e:
        error_msg = f"Error processing row {index}: {str(e)}"
        debug_log(error_msg)
        log_error(error_msg)
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
    total_rows = len(df)
    print("\nScanning for incomplete rows...")
    
    for index, row in tqdm(df.iterrows(), total=total_rows, desc="Scanning"):
        if is_incomplete_code(str(row['fixed_code'])):
            incomplete_count += 1
            debug_log(f"Row {index} identified as incomplete. Vulnerability type: {row['vulnerability_type']}")
    
    debug_log(f"Found {incomplete_count} incomplete rows out of {len(df)} total rows")
    print(f"\nFound {incomplete_count} incomplete rows out of {total_rows} total rows that need fixing...")
    
    # Get unprocessed rows
    unprocessed_rows = list(df.iloc[last_processed_index + 1:].iterrows())
    remaining_rows = len(unprocessed_rows)
    
    if remaining_rows == 0:
        print("No remaining rows to process. All done!")
        return
    
    print(f"\nProcessing {remaining_rows} remaining rows with {MAX_WORKERS} workers...")
    
    processed_count = 0
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Process rows in batches
        for batch_start in range(0, remaining_rows, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, remaining_rows)
            batch = unprocessed_rows[batch_start:batch_end]
            
            # Calculate progress
            progress = (batch_start / remaining_rows) * 100
            elapsed_time = time.time() - start_time
            rows_per_second = batch_start / elapsed_time if elapsed_time > 0 else 0
            estimated_remaining = (remaining_rows - batch_start) / rows_per_second if rows_per_second > 0 else 0
            
            progress_msg = (
                f"\nProgress: {progress:.1f}% ({batch_start}/{remaining_rows} rows)"
                f"\nProcessing speed: {rows_per_second:.1f} rows/second"
                f"\nEstimated time remaining: {estimated_remaining/3600:.1f} hours"
                f"\nProcessed so far: {processed_count} rows completed successfully"
                f"\nCurrently processing batch {batch_start//BATCH_SIZE + 1}, rows {batch_start} to {batch_end}"
            )
            print(progress_msg)
            debug_log(progress_msg)
            
            # Process batch
            futures = [executor.submit(process_row, row) for row in batch]
            
            # Handle results with progress bar
            completed_futures = 0
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc=f"Batch {batch_start//BATCH_SIZE + 1}"):
                try:
                    index, cleaned_code, is_valid = future.result()
                    if is_valid:
                        processed_df.at[index, 'fixed_code'] = cleaned_code
                        processed_df.to_csv(output_file, index=False)
                        update_progress(index)
                        processed_count += 1
                        debug_log(f"Successfully processed row {index}. Total processed: {processed_count}")
                    completed_futures += 1
                except Exception as e:
                    error_msg = f"Error processing batch: {str(e)}"
                    print(error_msg)
                    debug_log(error_msg)
            
            # Force garbage collection between batches
            gc.collect()
            
            # Check system resources
            if is_system_overloaded():
                debug_log("System resources high, pausing for 10 seconds...")
                print("\nSystem resources high, pausing for 10 seconds...")
                time.sleep(10)
    
    total_time = time.time() - start_time
    completion_msg = (
        f"\nProcessing complete!"
        f"\nTotal time: {total_time/3600:.1f} hours"
        f"\nProcessed {processed_count} rows out of {incomplete_count} incomplete rows"
        f"\nAverage speed: {processed_count/total_time:.1f} rows/second"
        f"\nUpdated file saved as {output_file}"
        f"\nCheck {debug_file} for detailed processing log"
    )
    debug_log(completion_msg)
    print(completion_msg)

if __name__ == "__main__":
    main() 