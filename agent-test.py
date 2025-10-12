"""
Custom Bank Statement Parser - Multi-Agent System
Integrates PDF parsing, analysis, code generation, and debugging agents.
"""

import os
import sys
import getpass
import argparse
import subprocess
import logging
import re
from pathlib import Path
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    filename="bank_parser_agent.log",
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# API KEY SETUP
# ============================================================================
def setup_api_keys():
    load_dotenv()
    """Setup required API keys for Groq and Google Gemini"""
    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
    
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# ============================================================================
# PDF PARSER CLASS
# ============================================================================
class PDFParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.pdf_text_dict = {}

    def check_format(self):
        """Check if the file is in a supported format (currently PDF only)."""
        if self.file_path.lower().endswith(".pdf"):
            return "pdf"
        else:
            return "unsupported"

    def parse_pdf(self):
        """
        Parses the PDF and stores text in a dictionary.
        Keys are page numbers (starting from 1), values are text.
        """
        try:
            pdf_reader = PdfReader(self.file_path)
            number_of_pages = len(pdf_reader.pages)

            for i in range(number_of_pages):
                page = pdf_reader.pages[i]
                text = page.extract_text()
                if text:
                    self.pdf_text_dict[i + 1] = text.strip()

            logger.info(f"✅ Successfully parsed {number_of_pages} pages from PDF")
            print(f"✅ Successfully parsed {number_of_pages} pages from PDF")

        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            print(f"❌ Error: File not found at {self.file_path}")
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            print(f"❌ Error reading PDF: {e}")

        return self.pdf_text_dict

    def run(self):
        file_format = self.check_format()
        if file_format == "pdf":
            return self.parse_pdf()
        else:
            print(f"⚠️ Unsupported file format: {file_format}")
            return {}

# ============================================================================
# ANALYSER AGENT CLASS
# ============================================================================
class AnalyserAgent:
    def __init__(self, pdf_text_dict):
        """
        Initialize the AnalyserAgent with parsed PDF text dictionary.
        Each key = page number, value = page text.
        """
        self.pdf_text_dict = pdf_text_dict
        self.analysis_results = {}

        # Initialize Groq LLM
        # self.groq_llm = ChatGroq(
        #     model="llama-3.3-70b-versatile",
        #     temperature=0,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        # )
        self.gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # Define the analysis prompt
        self.pdf_analysis_prompt = PromptTemplate(
            input_variables=["page_samples"],
            template=(
                "You are an expert in **PDF document structure analysis**, **financial data extraction**, and **Python-based text parsing**.\n\n"
"You are provided with raw text snippets extracted from a **bank statement PDF**.\n"
"Each snippet includes portions from the top, middle, and bottom of several pages:\n\n"
"{page_samples}\n\n"

"Treat all data as **plain text**, not numeric. Do not calculate or summarize values. "
"Words like 'Cr', 'Dr', 'Ltd', 'Balance', or similar terms are part of the document’s structure.\n\n"

"Your task has **two major goals**:\n\n"

"### PART 1 — Structural & Table Analysis\n"
"1. **File-Level Layout** — Identify how data flows (intro sections, transaction tables, footers, etc.).\n"
"2. **Table Structure Detection** — Identify recurring column headers strictly based on their textual representation.\n"
"3. **Pattern Consistency** — Comment on uniformity across pages: alignment issues, merged rows, line breaks within cells, or missing separators.\n"
"4. **Row & Cell Identification** — Describe how each transaction line can be detected (e.g., starts with a date pattern, followed by description and numeric values).\n"
"5. **Edge Cases & Challenges** — Predict parsing challenges such as irregular spacing, split lines, or OCR noise, and suggest regex or string-cleaning strategies.\n\n"

"### PART 2 — Parsing & Extraction Strategy\n"
"1. **Extraction Approach** — Recommend how to use **PyPDF2** for raw text extraction.\n"
"2. **Parsing Logic** — Explain how to detect transaction rows, split columns, and handle inconsistent delimiters.\n"
"3. **Output Schema** — Suggest ideal CSV output columns from the text.\n"
"4. **Error Handling** — Describe common pitfalls (like missing data, split lines, or merged balances) and ways to mitigate them.\n"
"5. **Extensibility** — Briefly mention how the code can later be extended for positional parsing or layout-based extraction if required.\n\n"

"Your response must be **precise**, **technical**, and **structured**, ending with a **Python code block** that demonstrates how text would be extracted using PyPDF2 (without parsing logic, only extraction setup)."

            )
        )

    def get_page_samples(self):
        """
        Build page samples with top, middle, and bottom text slices.
        """
        sample_texts = []
        for page, text in self.pdf_text_dict.items():
            n = len(text)
            samples = (
                f"Page {page}:\nTOP:\n{text[:300]}\n"
                f"MIDDLE:\n{text[n//2:n//2+100]}\n"
                f"BOTTOM:\n{text[-200:]}\n"
            )
            sample_texts.append(samples)
        return "\n\n".join(sample_texts)

    def analyze_pdf(self):
        """
        Run Groq LLM on the constructed samples and return structured analysis.
        """
        page_samples = self.get_page_samples()

        pdf_analysis_chain = self.pdf_analysis_prompt | self.gemini_llm
        response = pdf_analysis_chain.invoke({"page_samples": page_samples})

        self.analysis_results = response.content
        logger.info("✅ PDF structural analysis complete")
        return self.analysis_results

# ============================================================================
# CODING AGENT CLASS
# ============================================================================
class CodingAgent:
    def __init__(self, pdf_text_dict):
        """
        Step 1 — Use AnalyserAgent to understand PDF layout.
        Step 2 — Use Gemini to generate parsing code based on analysis.
        """
        self.pdf_text_dict = pdf_text_dict
        
        # Initialize Gemini LLM
        self.gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # Run structural analysis
        print("\n🔍 Running PDF structure analysis...")
        logger.info("Starting PDF structure analysis")
        analysis_agent = AnalyserAgent(pdf_text_dict=pdf_text_dict)
        self.analysis_report = analysis_agent.analyze_pdf()
        print("✅ Analysis complete\n")

        # Prepare code generation prompt
        self.pdf_code_prompt = PromptTemplate(
            input_variables=["analysis", "pdf_path"],
            template=(
                "You are an expert **Python developer for PDF-to-CSV conversion**.\n\n"
        "Below is a detailed **structural analysis** of a bank statement PDF:\n\n"
        "{analysis}\n\n"
        "Using this analysis, generate a **runnable Python script** that:\n"
        "1. Accepts a file path (provided as `{pdf_path}`)\n"
        "2. Extracts text using **PyPDF2** (mandatory)\n"
        "3. Parses transactions based on detected table patterns\n"
        "4. Exports all transactions to a CSV file named `output_transactions.csv`\n"
        "5. Includes logic to install missing dependencies automatically\n\n"
        "### Output Rules\n"
        "- Must include all imports and function definitions\n"
        "- Must be directly runnable (no missing parts)\n"
        "- Focus on clarity, maintainability, and easy debugging\n"
        "- Include your parsing logic with regex/string handling, not placeholders\n"
        "- Comment minimally, only where needed\n\n"
        "Return only the **complete Python script** — no explanations or text outside the code block."
            )
        )

    def generate_code(self, pdf_path: str):
        """
        Generates a runnable Python PDF parsing script using Gemini.
        """
        print("🔄 Generating parsing code using Gemini...")
        logger.info("Generating parsing code...")
        
        pdf_code_chain = self.pdf_code_prompt | self.gemini_llm
        response = pdf_code_chain.invoke({
            "analysis": self.analysis_report,
            "pdf_path": pdf_path
        })

        print("✅ Code generated\n")
        logger.info("Code generation complete")
        return response.content

# ============================================================================
# UTILITY: EXTRACT PYTHON CODE FROM MARKDOWN
# ============================================================================
def extract_python_code(response_text):
    """Extract Python code from markdown code blocks"""
    pattern = r'```python\s*(.*?)```'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()

# ============================================================================
# DEBUG AGENT CLASS
# ============================================================================
class PDFDebugAgent:
    def __init__(self, gemini_llm):
        """Initialize the debug agent with Gemini LLM"""
        self.debugAgent = gemini_llm
        self.max_attempts = 5
        self.script_path = Path("/mnt/a/Projects/KarbonAI_CodingAgent/ai-agent-challenge/custom_parsers/icici_parser.py")
        
    def debug_and_run(self, script_code, pdf_path=None):
        """
        Debug and execute the generated script with auto-fix capability.
        """
        attempt = 0
        success = False
        current_code = extract_python_code(script_code)
        
        logger.info("="*70)
        logger.info("Starting PDF Debug Agent")
        logger.info("="*70)
        
        while attempt < self.max_attempts and not success:
            attempt += 1
            logger.info(f"\nExecution Attempt {attempt}/{self.max_attempts}")
            print(f"\n{'='*70}")
            print(f"🔄 Execution Attempt {attempt}/{self.max_attempts}")
            print('='*70)
            
            # Write script to file
            self.script_path.write_text(current_code)
            logger.info(f"Script written to {self.script_path}")
            
            try:
                # Run the script
                print("▶️  Running script...")
                result = subprocess.run(
                    [sys.executable, str(self.script_path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env={**os.environ, "PDF_PATH": pdf_path or "sample.pdf"}
                )
                
                if result.returncode == 0:
                    logger.info(f"✅ SUCCESS on attempt {attempt}")
                    logger.info(f"Output:\n{result.stdout}")
                    print("✅ Script executed successfully!")
                    print("\n--- OUTPUT ---")
                    print(result.stdout)
                    success = True
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, 
                        result.args, 
                        result.stdout, 
                        result.stderr
                    )
            
            except subprocess.TimeoutExpired:
                error_msg = "Script execution timed out (>60s)"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                
                if attempt < self.max_attempts:
                    current_code = self._invoke_debug_agent(
                        current_code,
                        error_msg,
                        attempt
                    )
            
            except subprocess.CalledProcessError as e:
                error_output = e.stderr if e.stderr else e.stdout
                logger.error(f"Exit code {e.returncode}")
                logger.error(f"Error:\n{error_output}")
                print(f"❌ Execution failed (exit code {e.returncode})")
                print("\n--- ERROR ---")
                print(error_output)
                
                # Check for missing packages
                if "ModuleNotFoundError" in error_output or "ImportError" in error_output:
                    if self._install_missing_package(error_output):
                        print("✓ Package installed. Retrying...\n")
                        continue
                
                # Invoke debug agent for fixes
                if attempt < self.max_attempts:
                    current_code = self._invoke_debug_agent(
                        current_code,
                        error_output,
                        attempt
                    )
        
        if not success:
            logger.error("❌ All attempts failed. Manual intervention required.")
            print("\n" + "="*70)
            print("❌ Failed after all attempts")
            print("="*70)
            print(f"📋 Check bank_parser_agent.log for details")
            return False
        else:
            logger.info("✅ SUCCESS: PDF parsing script executed without errors")
            return True
    
    def _invoke_debug_agent(self, current_code, error_output, attempt):
        """Invoke Gemini to fix the code"""
        print("🤖 Invoking Debug Agent...")
        logger.info(f"Invoking debug agent (attempt {attempt})...")
        
        debug_prompt = f"""You are an expert Python debugger. Fix the following script that failed with an error.

**Current Script:**
```python
{current_code}
```

**Error Message:**
{error_output}

**Instructions:**
1. Analyze the error carefully
2. Fix all syntax errors, undefined variables, missing imports, typos, etc.
3. Ensure the script is production-ready and runnable
4. Add missing dependencies installation if needed
5. Return ONLY the complete fixed Python script in a ```python code block

**Important:** Return ONLY the code block, no explanations outside it."""
        
        response = self.debugAgent.invoke(debug_prompt)
        fixed_code = extract_python_code(response.content)
        
        logger.info(f"Debug agent response received")
        print("✓ Debug Agent fixed the script. Retrying...\n")
        
        return fixed_code
    
    def _install_missing_package(self, error_output):
        """Extract and install missing package"""
        try:
            missing_pkg = error_output.split("'")[1]
            logger.info(f"Missing package detected: {missing_pkg}")
            print(f"\n📦 Installing missing package: {missing_pkg}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", missing_pkg],
                capture_output=True
            )
            logger.info(f"Package {missing_pkg} installed")
            return True
        except (IndexError, subprocess.CalledProcessError) as e:
            logger.error(f"Failed to install package: {e}")
            return False

# ============================================================================
# MULTI-AGENT SYSTEM COORDINATOR
# ============================================================================
class CustomBankParserMultiAgentSystem:
    def __init__(self, file_path):
        self.file_path = file_path
        self.parser = PDFParser(file_path)
        self.pdf_text_dict = None
        self.coding_agent = None
        self.debug_agent = None

    def run_pipeline(self):
        """Execute the complete multi-agent pipeline"""
        print("="*70)
        print("CUSTOM BANK STATEMENT PARSER - MULTI-AGENT SYSTEM")
        print("="*70)
        
        # Step 1: Parse PDF
        print("\n📄 STEP 1: Parsing PDF file...")
        logger.info("="*70)
        logger.info("Starting Multi-Agent Pipeline")
        logger.info("="*70)
        
        self.pdf_text_dict = self.parser.run()
        
        if not self.pdf_text_dict:
            print("❌ Failed to parse PDF. Exiting.")
            logger.error("PDF parsing failed")
            return False
        
        # Step 2: Generate parsing code
        print("\n🤖 STEP 2: Analyzing PDF structure and generating code...")
        self.coding_agent = CodingAgent(pdf_text_dict=self.pdf_text_dict)
        generated_code = self.coding_agent.generate_code(pdf_path=self.file_path)
        
        # Step 3: Debug and execute
        print("\n🔧 STEP 3: Debugging and executing generated code...")
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        self.debug_agent = PDFDebugAgent(gemini_llm)
        success = self.debug_agent.debug_and_run(
            script_code=generated_code,
            pdf_path=self.file_path
        )
        
        if success:
            print("\n" + "="*70)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("="*70)
            print("📊 Check output_transactions.csv for results")
            logger.info("Pipeline completed successfully")
        else:
            print("\n" + "="*70)
            print("❌ PIPELINE FAILED")
            print("="*70)
            logger.error("Pipeline failed")
        
        return success

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Custom Bank Statement Parser - Multi-Agent System"
    )
    parser.add_argument("--target", required=True, help="Name of the bank (e.g., SBI, ICICI, HDFC)")
    parser.add_argument("--statement", help="Path to a sample statement file for testing")
    
    args = parser.parse_args()
    
    # Setup API keys
    setup_api_keys()
    
    # Validate file exists
    if not args.statement:
        bank_folder = os.path.join("data", args.target.lower())
        if os.path.isdir(bank_folder):
            pdf_files = [f for f in os.listdir(bank_folder) if f.lower().endswith(".pdf")]
            if pdf_files:
                args.statement = os.path.join(bank_folder, pdf_files[0])
                print(f"📄 Using PDF statement: {args.statement}")
            else:
                print(f"⚠️ No PDF files found in {bank_folder}.")
        else:
            print(f"⚠️ Folder not found: {bank_folder}")
    
    # Run the multi-agent system
    system = CustomBankParserMultiAgentSystem(file_path=args.statement)
    success = system.run_pipeline()

    sys.exit(0 if success else 1)