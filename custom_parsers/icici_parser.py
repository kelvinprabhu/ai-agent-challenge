import PyPDF2
import pandas as pd
import re
import os

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
    Parses transactions from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        pd.DataFrame: A DataFrame with the transaction data.
    """
    extracted_text = extract_text_from_pdf(pdf_path)
    if extracted_text is None:
        return None

    # Split the text into lines
    lines = extracted_text.split('\n')

    # Initialize lists to store transaction data
    dates = []
    descriptions = []
    debit_amts = []
    credit_amts = []
    balances = []

    # Initialize variables to store current transaction data
    current_date = None
    current_description = ""
    current_debit_amt = ""
    current_credit_amt = ""
    current_balance = ""

    # Iterate through the lines
    for line in lines:
        # Check if the line starts with a date pattern
        if re.match(r"^\d{2}-\d{2}-\d{4}", line):
            # If we have a current transaction, save it
            if current_date is not None:
                dates.append(current_date)
                descriptions.append(current_description.strip())
                debit_amts.append(current_debit_amt)
                credit_amts.append(current_credit_amt)
                balances.append(current_balance)

            # Extract the date and reset the current transaction data
            current_date = re.search(r"(\d{2}-\d{2}-\d{4})", line).group(1)
            current_description = ""
            current_debit_amt = ""
            current_credit_amt = ""
            current_balance = ""

            # Extract the description, debit/credit amount, and balance
            parts = line.split()
            for part in parts[1:]:
                if re.match(r"\d{1,3}(,\d{3})*(\.\d+)?", part):
                    if current_debit_amt == "":
                        current_debit_amt = part
                    elif current_credit_amt == "":
                        current_credit_amt = part
                    else:
                        current_balance = part
                else:
                    current_description += part + " "

        # If the line does not start with a date pattern, it's a continuation of the description
        else:
            current_description += line + " "

    # Save the last transaction
    if current_date is not None:
        dates.append(current_date)
        descriptions.append(current_description.strip())
        debit_amts.append(current_debit_amt)
        credit_amts.append(current_credit_amt)
        balances.append(current_balance)

    # Create a DataFrame with the transaction data
    df = pd.DataFrame({
        'Date': dates,
        'Description': descriptions,
        'Debit Amt': debit_amts,
        'Credit Amt': credit_amts,
        'Balance': balances
    })

    return df

if __name__ == "__main__":
    pdf_path = os.environ.get('PDF_PATH', 'data/icici/icici sample.pdf')
    df = parse(pdf_path)
    if df is not None:
        df.to_csv('output_transactions.csv', index=False)
        print(f"Successfully parsed {len(df)} transactions")
        print('Saved to output_transactions.csv')
    else:
        print("Failed to parse transactions")