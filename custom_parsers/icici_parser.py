import PyPDF2
import pandas as pd
import re
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parses a bank statement PDF to extract transaction data into a pandas DataFrame.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed transaction data.
                      Columns: ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
    """
    # 1. Extract text from PDF
    text = ""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n" # Add newline between pages for better line splitting
    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during PDF extraction: {e}")
        return pd.DataFrame()

    if not text:
        print("No text extracted from PDF.")
        return pd.DataFrame()

    # Regex patterns
    DATE_PATTERN = r"(\d{2}-\d{2}-\d{4})"
    AMOUNT_PATTERN = r"(\d{1,3}(?:,\d{3})*\.\d{2})" # e.g., 1,234.56 or 123.45

    transactions_data = []
    
    # 2. Pre-process lines to handle multi-line descriptions and split balances
    raw_lines = text.split('\n')
    processed_lines = []
    current_transaction_lines = []

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        # Check if the line starts with a date pattern, indicating a new transaction
        if re.match(DATE_PATTERN, line):
            if current_transaction_lines:
                # Join previous lines into a single logical transaction line
                processed_lines.append(" ".join(current_transaction_lines))
            current_transaction_lines = [line]
        else:
            # This line is a continuation of the previous transaction (description or split balance)
            current_transaction_lines.append(line)
    
    # Add the last transaction
    if current_transaction_lines:
        processed_lines.append(" ".join(current_transaction_lines))

    # 3. Parse each processed transaction line
    for full_line in processed_lines:
        # Specific pre-processing for "split balance" like "10 8.46" -> "108.46" at the end of the line.
        # This assumes the example "10\n8.46" means 108.46.
        # This regex looks for an integer followed by spaces and a decimal number at the end of the line.
        full_line = re.sub(r'(\d+)\s+(\d+\.\d{2})\s*$', r'\1\2', full_line)

        date_match = re.match(DATE_PATTERN, full_line)
        if not date_match:
            # This line doesn't start with a date, skip or log as an error
            continue

        date = date_match.group(1)
        
        # Extract the part of the line after the date
        remaining_text_after_date = full_line[date_match.end():].strip()

        # Extract Balance (always at the end)
        balance_match = re.search(AMOUNT_PATTERN + r'\s*$', remaining_text_after_date)
        balance = None
        if balance_match:
            balance = balance_match.group(1)
            # The text for description and debit/credit is everything before the balance
            remaining_text_for_dc_and_desc = remaining_text_after_date[:balance_match.start()]
        else:
            # If balance is not found, this line is malformed or not a transaction
            continue # Skip this line if balance is missing

        debit_amt = None
        credit_amt = None
        description = remaining_text_for_dc_and_desc.strip() # Default, will be refined

        # Find all potential amounts in the middle section (between date and balance)
        # Use finditer to get start/end positions, preserving original spacing for heuristic
        all_amount_matches = list(re.finditer(AMOUNT_PATTERN, remaining_text_for_dc_and_desc))

        if len(all_amount_matches) == 2:
            # If two amounts are found, assume the first is Debit and the second is Credit
            debit_match = all_amount_matches[0]
            credit_match = all_amount_matches[1]
            debit_amt = debit_match.group(1)
            credit_amt = credit_match.group(1)
            # Description is everything before the first amount
            description = remaining_text_for_dc_and_desc[:debit_match.start()].strip()
        elif len(all_amount_matches) == 1:
            # If one amount is found, use a heuristic based on trailing spaces to determine Debit/Credit
            single_amount_match = all_amount_matches[0]
            amount_value = single_amount_match.group(1)
            
            # Get the text segment before the amount, preserving original spacing
            text_before_amount = remaining_text_for_dc_and_desc[:single_amount_match.start()]
            
            # Count trailing spaces before the amount
            num_trailing_spaces = len(text_before_amount) - len(text_before_amount.rstrip(' '))
            
            # Heuristic: If there are few trailing spaces (e.g., < 5), it's likely Debit.
            # Otherwise (more spaces), it's likely Credit.
            # This threshold is an educated guess based on typical PDF text extraction and sample output.
            # It aims to distinguish between "Description Amount" (Debit) and "Description    Amount" (Credit).
            if num_trailing_spaces < 5: # Arbitrary threshold, adjust if needed
                debit_amt = amount_value
            else:
                credit_amt = amount_value
            
            description = text_before_amount.strip() # Clean description after determining amount type
        else:
            # No amounts found before balance, so description is the whole remaining_text_for_dc_and_desc
            description = remaining_text_for_dc_and_desc.strip()

        # Clean up description (remove extra spaces, leading/trailing non-alphanumeric)
        description = re.sub(r'\s+', ' ', description).strip()
        description = re.sub(r'^[^\w\s]+', '', description).strip() # Remove leading non-alphanumeric
        description = re.sub(r'[^\w\s]+$', '', description).strip() # Remove trailing non-alphanumeric

        transactions_data.append({
            'Date': date,
            'Description': description,
            'Debit Amt': debit_amt,
            'Credit Amt': credit_amt,
            'Balance': balance
        })

    # Create DataFrame
    df = pd.DataFrame(transactions_data)
    
    # Replace None with NaN for pandas
    df = df.replace({None: pd.NA})

    return df

if __name__ == "__main__":
    # Determine PDF path
    pdf_path = os.environ.get('PDF_PATH', 'data/icici/icici sample.pdf')

    # Create dummy PDF for testing if the default path doesn't exist
    if not os.path.exists(pdf_path):
        print(f"Warning: Default PDF path '{pdf_path}' not found. Creating a dummy PDF for demonstration.")
        os.makedirs(os.path.dirname(pdf_path) or '.', exist_ok=True)
        
        # Create a simple dummy PDF with content matching the analysis
        # This dummy content is designed to test the parsing logic, especially the
        # heuristic for distinguishing Debit/Credit based on spacing.
        dummy_content = """
01-08-2024 Salary Credit XYZ Pvt Ltd 1935.30 6864.58
02-08-2024 Salary Credit XYZ Pvt Ltd        1652.61 8517.19
03-08-2024 IMPS UPI Payment Amazon 3886.08 4631.11
04-08-2024 Mobile Recharge Via UPI 500.00 4131.11
05-08-2024 Interest Credit        100.50 4231.61
06-08-2024 ATM Withdrawal 2000.00 2231.61
07-08-2024 Bill Payment Electricity 750.25 1481.36
08-08-2024 Refund from Merchant        250.00 1731.36
09-08-2024 Rent Payment 5000.00 108.46
10-08-2024 Bank Charges 10.00 98.46
11-08-2024 Split Balance Example 10 8.46 # This should become 108.46 due to pre-processing
ChatGPT Powered Karbon Bannk
        """
        try:
            c = canvas.Canvas(pdf_path, pagesize=letter)
            textobject = c.beginText()
            textobject.setTextOrigin(50, 750)
            textobject.setFont("Helvetica", 10)
            for line in dummy_content.strip().split('\n'):
                textobject.textLine(line)
            c.drawText(textobject)
            c.save()
            print(f"Dummy PDF created at '{pdf_path}' for testing.")
        except ImportError:
            print("ReportLab not installed. Cannot create dummy PDF. Please install it (`pip install reportlab`) or provide a real PDF.")
            print("Exiting as PDF is required for parsing.")
            exit(1)


    # Parse the PDF
    transactions_df = parse(pdf_path)

    if not transactions_df.empty:
        output_csv_path = 'output_transactions.csv'
        transactions_df.to_csv(output_csv_path, index=False)
        print(f"Successfully parsed {len(transactions_df)} transactions.")
        print(f"Saved to {output_csv_path}")
    else:
        print("No transactions parsed.")