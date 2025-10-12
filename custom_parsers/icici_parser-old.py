import pandas as pd
import PyPDF2
import re
import subprocess

def install_missing_dependencies():
    try:
        import pandas
        import PyPDF2
    except ImportError:
        print("Installing missing dependencies...")
        subprocess.check_call(['pip', 'install', 'pandas', 'PyPDF2'])
        print("Dependencies installed.")

install_missing_dependencies()

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyPDF2.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    return text

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parses a bank statement PDF and extracts transaction data into a pandas DataFrame.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        pd.DataFrame: A DataFrame containing the transaction data.
    """
    extracted_text = extract_text_from_pdf(pdf_path)

    if not extracted_text:
        return pd.DataFrame()

    lines = extracted_text.splitlines()
    transactions = []
    current_transaction = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        date_match = re.match(r"^\d{2}-\d{2}-\d{4}", line)
        if date_match:
            if current_transaction:
                transactions.append(current_transaction)

            current_transaction = {
                'Date': date_match.group(0),
                'Description': '',
                'Debit Amt': None,
                'Credit Amt': None,
                'Balance': None
            }
            parts = line.split()
            date_str = parts[0]
            remaining_line = line[len(date_str):].strip()

            # Regex to find amounts
            amount_matches = list(re.finditer(r"(\d{1,3}(?:,\d{3})*\.\d{2})|(\d+\.\d{2})", remaining_line))

            if len(amount_matches) >= 1:
                first_amount_start = amount_matches[0].start()
                current_transaction['Description'] = remaining_line[:first_amount_start].strip()

                if len(amount_matches) >= 2:
                    second_amount_start = amount_matches[1].start()
                    first_amount = remaining_line[first_amount_start:second_amount_start].strip()
                    second_amount = remaining_line[second_amount_start:].strip()

                    # Determine Debit/Credit based on order
                    if current_transaction['Debit Amt'] is None:
                        current_transaction['Debit Amt'] = first_amount
                        current_transaction['Credit Amt'] = second_amount
                    else:
                        current_transaction['Credit Amt'] = first_amount
                        current_transaction['Balance'] = second_amount
                else:
                    first_amount = remaining_line[first_amount_start:].strip()
                    if current_transaction['Debit Amt'] is None:
                        current_transaction['Debit Amt'] = first_amount
                    else:
                        current_transaction['Balance'] = first_amount
            else:
                current_transaction['Description'] = remaining_line.strip()

        elif current_transaction:
            # Handle split lines in Description or Balance
            if current_transaction['Description']:
                current_transaction['Description'] += " " + line
            else:
                current_transaction['Description'] = line

    if current_transaction:
        transactions.append(current_transaction)

    df = pd.DataFrame(transactions)
    return df

if __name__ == "__main__":
    pdf_file_path = 'bank_statement.pdf'  # Replace with your PDF file path
    try:
        df = parse(pdf_file_path)
        df.to_csv('output_transactions.csv', index=False)
        print("Successfully parsed and saved transactions to output_transactions.csv")
    except Exception as e:
        print(f"An error occurred during parsing: {e}")