import subprocess
import sys
import re
import csv
import io
import os

# --- Dependency Installation ---
def install_package(package):
    """
    Installs a Python package if it's not already installed.
    """
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} installed successfully.")

# Ensure PyPDF2 and pandas are installed
install_package("PyPDF2")
install_package("pandas")

# Now import the installed packages
import PyPDF2
import pandas as pd

# --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_path):
    """
    Extracts raw text from each page of a PDF using PyPDF2.
    Returns a list of strings, where each string is the text content of a page.
    """
    all_page_text = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    all_page_text.append(text)
    except Exception as e:
        print(f"Error extracting text from PDF '{pdf_path}': {e}")
    return all_page_text

# --- Parsing Logic ---
def parse_bank_statement_text(page_texts):
    """
    Parses the raw text from bank statement pages into structured transaction data.
    Implements a two-pass approach for line merging and field extraction.
    """
    all_cleaned_lines = []
    header_pattern = r"Date Description Debit Amt Credit Amt Balance"
    footer_pattern = r"ChatGPT Powered Karbon Bannk"

    # Pass 0: Initial cleaning and flattening of page texts into a single list of lines
    for page_text in page_texts:
        lines = page_text.split('\n')
        for line in lines:
            stripped_line = line.strip()
            if stripped_line and \
               not re.fullmatch(header_pattern, stripped_line) and \
               not re.fullmatch(footer_pattern, stripped_line):
                all_cleaned_lines.append(stripped_line)

    merged_lines = []
    current_buffer = ""
    date_start_regex = r"^\d{2}-\d{2}-\d{4}"
    numeric_continuation_regex = r"^\.?\d" # Starts with . or a digit
    ends_with_two_numbers_regex = r"\d+\.?\d*\s+\d+\.?\d*$" # Ends with two numbers
    
    # Helper to check if a buffer is "complete enough" to be finalized
    # A buffer is considered complete if it has a date and at least one number,
    # or if it contains two numbers at the end (implying a missing date/description transaction).
    def is_buffer_complete(buffer):
        if not buffer:
            return False
        if re.search(date_start_regex, buffer) and re.search(r"\d+\.?\d*", buffer):
            return True
        if re.search(ends_with_two_numbers_regex, buffer):
            return True
        return False

    # Pass 1: Line Merging
    for line in all_cleaned_lines:
        # Heuristic 1: New transaction starts with a date
        if re.match(date_start_regex, line):
            if current_buffer:
                merged_lines.append(current_buffer)
            current_buffer = line
        # Heuristic 3: Numeric continuation (e.g., "101" + ".59")
        elif re.match(numeric_continuation_regex, line) and current_buffer and re.search(r"\d+\.?\d*$", current_buffer):
            current_buffer = re.sub(r"(\d+\.?\d*)$", r"\1" + line, current_buffer)
        # Heuristic 2: Line itself looks like a complete transaction (ends with two numbers)
        # This should always start a new transaction if the current buffer is already complete.
        elif re.search(ends_with_two_numbers_regex, line):
            if current_buffer and is_buffer_complete(current_buffer):
                merged_lines.append(current_buffer)
                current_buffer = line
            elif current_buffer: # If buffer exists but not complete, append to it (e.g., multi-line description ending with numbers)
                current_buffer += " " + line
            else: # Buffer is empty, this is the first line, treat as new transaction
                current_buffer = line
        # Default: General continuation (e.g., multi-line description or unhandled fragment like "31-")
        else:
            # If current_buffer is already complete, and this line is not a continuation of numbers,
            # then this line must be the start of a new transaction's description.
            if current_buffer and is_buffer_complete(current_buffer):
                merged_lines.append(current_buffer)
                current_buffer = line # Start new buffer with this line (e.g., "31-")
            else:
                current_buffer += " " + line
    
    if current_buffer:
        merged_lines.append(current_buffer)

    # Pass 2: Field Extraction
    transactions = []
    last_known_date = None
    
    # Regex to capture optional date, description, and the two final numeric values
    transaction_line_regex = re.compile(
        r"^(?P<date>\d{2}-\d{2}-\d{4})?\s*"  # Optional Date
        r"(?P<description>.*?)\s*"          # Description (non-greedy, optional trailing space)
        r"(?P<transaction_amt>\d+\.?\d*)"   # Transaction Amount (Debit/Credit)
        r"(?:\s+(?P<balance>\d+\.?\d*))?$"  # Optional Balance
    )

    for m_line in merged_lines:
        match = transaction_line_regex.search(m_line)
        if match:
            data = match.groupdict()
            
            date = data['date']
            description = data['description'].strip()
            transaction_amt = data['transaction_amt']
            balance = data['balance'] if data['balance'] else "" # Ensure balance is empty string if None

            # Handle missing date by propagating last known date
            if date is None:
                date = last_known_date
            else:
                last_known_date = date

            withdrawals = ""
            deposits = ""

            # Heuristic to determine if it's a withdrawal or deposit based on description keywords
            desc_lower = description.lower()
            if "salary credit" in desc_lower or \
               ("credit" in desc_lower and "card" not in desc_lower and "payment" not in desc_lower) or \
               "deposit" in desc_lower:
                deposits = transaction_amt
            else:
                withdrawals = transaction_amt
            
            transactions.append({
                'Date': date,
                'Description': description,
                'Withdrawals': withdrawals,
                'Deposits': deposits,
                'Balance': balance
            })
        else:
            # Log unparseable lines for debugging
            print(f"Warning: Could not parse transaction line: {m_line}")

    return transactions

# --- Main Execution ---
if __name__ == "__main__":
    pdf_file_path = "data/icici/icici sample.pdf" # Specified PDF file path
    output_csv_path = "output_transactions.csv"

    # Create dummy PDF file for demonstration if it doesn't exist
    # In a real scenario, this file would be provided.
    if not os.path.exists(pdf_file_path):
        os.makedirs(os.path.dirname(pdf_file_path), exist_ok=True)
        # Using the simulated raw_page_texts_simulated from the prompt
        # to ensure the parsing logic works as intended with the provided examples.
        print(f"'{pdf_file_path}' not found. Using simulated text for parsing demonstration.")
        raw_page_texts_simulated = [
            """Date Description Debit Amt Credit Amt Balance
01-08-2024 Salary Credit XYZ Pvt Ltd 1935.3 6864.58
02-08-2024 Salary Credit XYZ Pvt Ltd 1652.61 8517.19
03-08-2024 IMPS UPI Payment Amazon 3886.08 4631.11
03-08-2024 Mobile Recharge Via UPI 1648.72 6279.83
14-08-2024 Fuel Purchase Debit Card 3878.57 101
.59
26-10-2024 EMI Auto Debit HDFC Bank 1006.21 16630.38
04-11-2024 Service Charge GST Debit 756.93 
 3782.46 6419.93
23-01-2025 Credit Card Payment ICICI 426.36 6846.29
27-01-2025 Service Charge GST Debit 4332.26 2514.03
27-01-2025 Fuel Purchase Debit Card 1533.65 4047.68ChatGPT Powered Karbon Bannk""",

            """Date Description Debit Amt Credit Amt Balance
30-01-2025 UPI QR Payment Groceries 4960.86 9008.54
02-02-2025 IMPS UPI Payment Amazon 2693.97 11702.51
14-02-2025 Online Card Purchase Flipkart 737.74 12440.25
21-02-2025 Dining Out Card Swipe 3973.65 8466.6
24-02-2025 IMPS UPI Payment Amazon 1998.34 10
25 Salary Credit XYZ Pvt Ltd 1863.31 10587.99
21-05-2025 Fuel Purchase Debit Card 4526.6 6061.39
31-
8.46 6914.6
24-07-2025 Electricity Bill NEFT Online 2917.52 3997.08
25-07-2025 Salary Credit XYZ Pvt Ltd 566.32 3430.76
27-07-2025 ATM Cash Withdrawal India 2156.01 5586.77ChatGPT Powered Karbon Bannk"""
        ]
        extracted_texts = raw_page_texts_simulated
    else:
        extracted_texts = extract_text_from_pdf(pdf_file_path)

    if extracted_texts:
        parsed_transactions = parse_bank_statement_text(extracted_texts)

        if parsed_transactions:
            df = pd.DataFrame(parsed_transactions)
            df.to_csv(output_csv_path, index=False)
            print(f"Successfully extracted {len(parsed_transactions)} transactions to '{output_csv_path}'")
        else:
            print("No transactions parsed.")
    else:
        print("No text extracted from PDF. Check file path or PDF content.")