# Complete Setup Guide

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning repository)
- API keys for Groq and Google Gemini

## 🚀 Step-by-Step Setup

### 1. Clone Repository

```bash
git clone https://github.com/apurv-korefi/ai-agent-challenge.git
cd ai-agent-challenge
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup API Keys

#### Option A: Environment Variables

```bash
export GROQ_API_KEY="your-groq-api-key-here"
export GOOGLE_API_KEY="your-google-api-key-here"
```

#### Option B: `.env` File (Recommended)

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your-groq-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

**Get API Keys:**
- **Groq**: https://console.groq.com/keys
- **Google Gemini**: https://makersuite.google.com/app/apikey

### 5. Prepare Data Structure

Create the following folder structure:

```
ai-agent-challenge/
├── data/
│   ├── icici/
│   │   ├── icici_sample.pdf          # Your ICICI bank statement
│   │   └── icici_expected.csv        # Expected parsed output
│   ├── sbi/
│   │   ├── sbi_sample.pdf
│   │   └── sbi_expected.csv
│   └── hdfc/
│       ├── hdfc_sample.pdf
│       └── hdfc_expected.csv
```

**Create directories:**

```bash
mkdir -p data/icici data/sbi data/hdfc
```

### 6. Prepare Expected CSV Format

Your expected CSV should match the transaction structure. Example format:

```csv
date,description,debit,credit,balance,reference
2024-01-15,ATM Withdrawal,5000.00,,45000.00,ATM123456
2024-01-16,Salary Credit,,50000.00,95000.00,SAL202401
2024-01-17,Online Transfer,2000.00,,93000.00,NEFT789012
```

**Common column names:**
- `date` or `transaction_date`
- `description` or `particulars` or `narration`
- `debit` or `withdrawal` or `debit_amount`
- `credit` or `deposit` or `credit_amount`
- `balance` or `closing_balance`
- `reference` or `cheque_no` or `ref_no` (optional)

## 🧪 Testing the Setup

### Test 1: Basic Run

```bash
python agent.py --target icici
```

**Expected output:**
```
======================================================================
🚀 CUSTOM BANK PARSER - LANGGRAPH MULTI-AGENT SYSTEM
======================================================================

======================================================================
📄 NODE 1: PDF PARSING
======================================================================
✅ Successfully parsed 4 pages
...
```

### Test 2: Validate Parser

```bash
python test_parser.py --bank icici
```

**Expected output:**
```
======================================================================
📊 VALIDATION REPORT
======================================================================

✅ SUMMARY
  • Exact Match: YES ✅
  • Cell Match: 100.00%
  • Matching Cells: 270/270
...
```

### Test 3: Check Generated Files

```bash
# Check if parser was created
ls -l custom_parsers/icici_parser.py

# Check if output CSV was created
ls -l output_transactions.csv

# View parser code
cat custom_parsers/icici_parser.py

# View parsed output
head output_transactions.csv
```

## 🔧 Configuration Options

### Agent Configuration

```bash
# Basic usage
python agent.py --target <bank_name>

# With custom PDF
python agent.py --target icici --statement path/to/statement.pdf

# With expected CSV for validation
python agent.py --target icici \
  --statement data/icici/icici_sample.pdf \
  --expected-csv data/icici/icici_expected.csv

# With more retry attempts
python agent.py --target sbi --max-attempts 5
```

### Test Configuration

```bash
# Basic validation
python test_parser.py --bank icici

# With custom files
python test_parser.py --bank icici \
  --pdf data/icici/custom.pdf \
  --expected data/icici/custom_expected.csv

# Strict mode (fail if not 100% match)
python test_parser.py --bank icici --strict
```

## 📝 Creating Your Own Bank Parser

### Step 1: Prepare Your Data

1. Get a sample bank statement PDF
2. Manually create the expected CSV output
3. Place both in `data/<bank_name>/`

```bash
mkdir -p data/mybank
# Copy your files:
# - mybank_sample.pdf
# - mybank_expected.csv
```

### Step 2: Run the Agent

```bash
python agent.py --target mybank
```

### Step 3: Validate Output

```bash
python test_parser.py --bank mybank
```

### Step 4: Review Generated Parser

```bash
cat custom_parsers/mybank_parser.py
```

### Step 5: Use the Parser

```python
from custom_parsers.mybank_parser import parse

# Parse a new statement
df = parse("path/to/new/statement.pdf")
print(df.head())

# Save to CSV
df.to_csv("parsed_output.csv", index=False)
```

## 🐛 Troubleshooting

### Issue: Module Not Found Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue: API Key Errors

```bash
# Verify API keys are set
echo $GROQ_API_KEY
echo $GOOGLE_API_KEY

# Or check .env file
cat .env
```

### Issue: PDF Parsing Fails

**Possible causes:**
1. Encrypted PDF - remove password protection
2. Scanned image PDF - needs OCR preprocessing
3. Corrupted PDF - try re-downloading

**Solutions:**
```bash
# Check PDF validity
python -c "from PyPDF2 import PdfReader; reader = PdfReader('data/icici/icici_sample.pdf'); print(f'Pages: {len(reader.pages)}')"
```

### Issue: Low Matching Percentage

**Debug steps:**

1. Check the generated parser:
```bash
cat custom_parsers/icici_parser.py
```

2. Check the parsed output:
```bash
head -20 output_transactions.csv
```

3. Compare with expected:
```bash
head -20 data/icici/icici_expected.csv
```

4. Review detailed logs:
```bash
tail -100 bank_parser_agent.log
```

5. Try with more attempts:
```bash
python agent.py --target icici --max-attempts 5
```

### Issue: Code Generation Timeout

```bash
# The agent uses Gemini 2.0 Flash which is very fast
# If timeout occurs, check your internet connection
# Or try running again - might be API rate limiting
```

## 📊 Expected Results

### Success Metrics

- **Parsing Success**: Parser executes without errors
- **Validation Score**: 
  - 100% = Perfect match ✅
  - 95-99% = Excellent (minor formatting differences) ✅
  - 80-94% = Good (needs review) ⚠️
  - <80% = Needs fixing ❌

### Generated Artifacts

After successful run, you should have:

1. **Parser Script**: `custom_parsers/<bank>_parser.py`
   - Contains `parse(pdf_path: str) -> pd.DataFrame` function
   - Fully runnable and reusable

2. **Output CSV**: `output_transactions.csv`
   - Parsed transaction data
   - Matches expected schema

3. **Log File**: `bank_parser_agent.log`
   - Detailed execution logs
   - Useful for debugging

## 🎓 Advanced Usage

### Batch Processing

```python
# Create a batch processing script
import os
from custom_parsers.icici_parser import parse

pdf_folder = "statements/icici/"
output_folder = "parsed_statements/"

for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        df = parse(pdf_path)
        
        output_name = pdf_file.replace(".pdf", ".csv")
        output_path = os.path.join(output_folder, output_name)
        df.to_csv(output_path, index=False)
        
        print(f"✅ Processed: {pdf_file}")
```

### Custom Validation

```python
import pandas as pd
from custom_parsers.icici_parser import parse

# Parse statement
df = parse("data/icici/icici_sample.pdf")

# Custom validation rules
assert len(df) > 0, "No transactions found"
assert all(df['balance'] >= 0), "Negative balances found"
assert df['date'].is_monotonic_increasing, "Dates not in order"

# Business logic checks
total_credits = df['credit'].sum()
total_debits = df['debit'].sum()
print(f"Total Credits: ₹{total_credits:,.2f}")
print(f"Total Debits: ₹{total_debits:,.2f}")
print(f"Net Change: ₹{(total_credits - total_debits):,.2f}")
```

### Integration with Other Systems

```python
# Example: Upload to database
import pandas as pd
from sqlalchemy import create_engine
from custom_parsers.icici_parser import parse

# Parse statement
df = parse("data/icici/statement_jan2024.pdf")




## 📚 Additional Resources

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **LangChain Documentation**: https://python.langchain.com/docs/
- **Groq Documentation**: https://console.groq.com/docs
- **Google Gemini API**: https://ai.google.dev/docs

## 🤝 Support

If you encounter issues:

1. Check `bank_parser_agent.log` for detailed error messages
2. Review the troubleshooting section above
3. Ensure all dependencies are correctly installed
4. Verify API keys are valid and have sufficient quota

---

**Ready to start?** Follow the steps above and run:

```bash
python agent.py --target icici
```

Good luck! 🚀