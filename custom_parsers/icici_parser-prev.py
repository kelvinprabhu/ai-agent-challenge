import subprocess
import sys
import re
import pandas as pd
from datetime import datetime, timedelta
import os

# --- Dependency Installation ---
def install_missing_dependencies():
    """
    Checks for and installs required Python packages.
    """
    required_packages = ['PyPDF2', 'pandas']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"{package} installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")
                print("Please install it manually using: pip install {package}")
                sys.exit(1) # Exit if a critical dependency cannot be installed

install_missing_dependencies()

# Now import after ensuring they are installed
import PyPDF2

# --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_path):
    """
    Extracts raw text content from each page of a PDF document.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        list: A list of strings, where each string is the raw text content
              of a page. Returns an empty list if the file cannot be read.
    """
    page_texts = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_texts.append(page.extract_text())
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
    except PyPDF2.errors.PdfReadError:
        print(f"Error: Could not read PDF file {pdf_path}. It might be corrupted or encrypted.")
    except Exception as e:
        print(f"An unexpected error occurred during PDF extraction: {e}")
    return page_texts

# --- Parsing Logic ---
def parse_bank_statement(pdf_text_pages):
    """
    Parses transaction data from a list of raw text pages extracted from a bank statement PDF.

    Args:
        pdf_text_pages (list): A list of strings, each representing the text of a PDF page.

    Returns:
        list: A list of dictionaries, where each dictionary represents a parsed transaction.
    """
    all_raw_lines = []
    for page_text in pdf_text_pages:
        all_raw_lines.extend(page_text.split('\n'))

    # Pre-process lines: strip whitespace, normalize internal spaces, filter headers/footers
    cleaned_lines = []
    header_pattern = re.compile(r"Date\s+Description\s+Debit\s+Amt\s+Credit\s+Amt\s+Balance")
    footer_pattern = re.compile(r"ChatGPT\s+Powered\s+Karbon\s+Bannk")

    for line in all_raw_lines:
        line = line.strip()
        line = re.sub(r'\s+', ' ', line) # Normalize internal whitespace
        if line and not (header_pattern.fullmatch(line) or footer_pattern.fullmatch(line)):
            cleaned_lines.append(line)

    # Assemble multi-line transactions into single strings
    assembled_lines = []
    current_line_buffer = []
    
    # Regex to detect start of a new transaction:
    # 1. Full date (DD-MM-YYYY) or partial date (DD) followed by text
    date_led_transaction_start_pattern = re.compile(r"^\d{2}(?:-\d{2}-\d{4})?\s+[A-Za-z]")
    # 2. Line starting with a number, followed by another number (e.g., "3782.46 6419.93")
    # This is for lines that are transactions but lack a date.
    number_led_transaction_start_pattern = re.compile(r"^\d+\.?\d*\s+\d+\.?\d*$")

    for line in cleaned_lines:
        if line.startswith('.'): # Numeric continuation like ".59"
            if current_line_buffer:
                # Append directly to the last part of the buffer to complete a number
                current_line_buffer[-1] += line
            else:
                # This case should ideally not happen if the first line is always a transaction start
                current_line_buffer.append(line)
        elif date_led_transaction_start_pattern.match(line) or number_led_transaction_start_pattern.match(line):
            if current_line_buffer:
                assembled_lines.append(" ".join(current_line_buffer))
            current_line_buffer = [line]
        else: # Description continuation or other non-start line
            current_line_buffer.append(line)
    
    # Add any remaining lines in the buffer after the loop
    if current_line_buffer:
        assembled_lines.append(" ".join(current_line_buffer))

    transactions_data = []
    last_full_date_obj = None

    # Regex to parse an assembled transaction line
    # This pattern captures the date (if present) and the rest of the line.
    # The rest of the line will then be parsed for description and amounts.
    main_line_pattern = re.compile(
        r"^(?P<date_part>\d{2}(?:-\d{2}-\d{4})?)?\s*(?P<rest_of_line>.+)$"
    )

    # Regex to extract amounts from the end of 'rest_of_line'
    # This looks for one or two numbers at the end, and the preceding text as description.
    amounts_pattern = re.compile(
        r"(?P<description_core>.+?)\s+" # Non-greedy description
        r"(?P<amount_val1>\d+\.?\d*)"   # First amount (transaction amount or balance)
        r"(?:\s+(?P<amount_val2>\d+\.?\d*))?$" # Optional second amount (balance)
    )

    deposit_keywords = ["salary", "credit", "cr", "deposit"]
    withdrawal_keywords = ["debit", "payment", "purchase", "withdrawal", "charge", "emi", "dr"]

    for assembled_line in assembled_lines:
        main_match = main_line_pattern.match(assembled_line)
        if not main_match:
            print(f"Warning: Main line pattern did not match: {assembled_line}")
            continue

        date_part = main_match.group('date_part')
        rest_of_line = main_match.group('rest_of_line').strip()

        amounts_match = amounts_pattern.match(rest_of_line)
        if not amounts_match:
            print(f"Warning: Amounts pattern did not match for rest of line: {rest_of_line} (from {assembled_line})")
            continue
        
        data = amounts_match.groupdict()
        description = data['description_core'].strip()
        amount_val1_str = data['amount_val1']
        amount_val2_str = data['amount_val2']

        # Initialize parsed values
        current_date_str = date_part # Default to original string
        withdrawals = 0.0
        deposits = 0.0
        balance = None
        transaction_amount = None

        # --- Date Inference ---
        if re.fullmatch(r"\d{2}-\d{2}-\d{4}", date_part or ""):
            try:
                current_date_obj = datetime.strptime(date_part, '%d-%m-%Y')
                last_full_date_obj = current_date_obj
                current_date_str = current_date_obj.strftime('%d-%m-%Y')
            except ValueError:
                pass # current_date_str remains date_part if parsing fails
        elif re.fullmatch(r"\d{2}", date_part or "") and last_full_date_obj:
            try:
                day = int(date_part)
                inferred_month = last_full_date_obj.month
                inferred_year = last_full_date_obj.year

                # Heuristic: if day is less than last known day, assume next month
                if day < last_full_date_obj.day:
                    inferred_month += 1
                    if inferred_month > 12:
                        inferred_month = 1
                        inferred_year += 1
                
                inferred_date_obj = datetime(inferred_year, inferred_month, day)
                current_date_str = inferred_date_obj.strftime('%d-%m-%Y')
            except ValueError:
                pass # current_date_str remains date_part if inference fails

        # --- Amount Assignment ---
        try:
            # Clean amount strings (remove non-numeric except dot)
            amount_val1_str = re.sub(r'[^\d.]', '', amount_val1_str)
            amount_val2_str = re.sub(r'[^\d.]', '', amount_val2_str) if amount_val2_str else None

            transaction_amount = float(amount_val1_str)
            if amount_val2_str:
                balance = float(amount_val2_str)
            
            # Special handling for lines like "3782.46 6419.93" where date is missing
            # and the 'description' field from regex is actually an amount.
            if date_part is None and re.fullmatch(r"\d+\.?\d*", description):
                # This means 'description' is actually the transaction amount
                # and 'transaction_amount' (amount_val1_str) is the balance.
                withdrawals = float(description) # Assuming it's a withdrawal if no description
                balance = transaction_amount # The second number is the balance
                description = "Unknown Transaction (Date Missing)" # Default description
            else:
                # Normal case: description is text, amount_val1 is transaction amount
                lower_description = description.lower()
                is_deposit = any(kw in lower_description for kw in deposit_keywords)
                is_withdrawal = any(kw in lower_description for kw in withdrawal_keywords)

                if is_deposit and not is_withdrawal:
                    deposits = transaction_amount
                elif is_withdrawal and not is_deposit:
                    withdrawals = transaction_amount
                elif is_deposit and is_withdrawal: # Ambiguous, e.g., "Credit Card Payment"
                    # Default to withdrawal for "Payment" or "Debit"
                    if "payment" in lower_description or "debit" in lower_description:
                        withdrawals = transaction_amount
                    else: # Default to deposit if still ambiguous
                        deposits = transaction_amount
                else: # No clear keywords, default to withdrawal
                    withdrawals = transaction_amount

        except ValueError:
            print(f"Warning: Could not parse numeric amount(s) for line: {assembled_line}. Amounts set to 0/None.")
            # Keep amounts as 0.0 or None if parsing fails

        transactions_data.append({
            'Date': current_date_str,
            'Description': description,
            'Withdrawals': withdrawals,
            'Deposits': deposits,
            'Balance': balance
        })
    
    return transactions_data

# --- Main Execution Block ---
if __name__ == "__main__":
    pdf_file_path = 'data/icici/icici sample.pdf' # User-provided path
    output_csv_path = 'output_transactions.csv'

    # Ensure the directory for the PDF exists if it's a relative path
    if not os.path.exists(pdf_file_path) and not os.path.isabs(pdf_file_path):
        print(f"Error: PDF file not found at '{pdf_file_path}'. Please ensure the path is correct.")
        sys.exit(1)

    # 1. Extract text from PDF
    extracted_pages_text = extract_text_from_pdf(pdf_file_path)

    if not extracted_pages_text:
        print("No text extracted from PDF. Exiting.")
    else:
        # 2. Parse transactions from extracted text
        transactions_data = parse_bank_statement(extracted_pages_text)

        if transactions_data:
            # 3. Export to CSV
            df = pd.DataFrame(transactions_data)
            # Define column order for CSV
            df = df[['Date', 'Description', 'Withdrawals', 'Deposits', 'Balance']]
            df.to_csv(output_csv_path, index=False)
            print(f"Successfully extracted {len(transactions_data)} transactions to {output_csv_path}")
        else:
            print("No transactions parsed.")