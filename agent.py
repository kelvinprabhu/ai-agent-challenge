"""
Custom Bank Statement Parser - Multi-Agent System with LangGraph
Integrates PDF parsing, analysis, code generation, debugging, and validation agents.
"""

import os
import sys
import getpass
import argparse
import subprocess
import logging
import re
import pandas as pd
from pathlib import Path
from typing import TypedDict
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
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
# STATE DEFINITION FOR LANGGRAPH
# ============================================================================
class AgentState(TypedDict):
    """State that flows through the agent graph"""
    pdf_path: str
    bank_name: str
    csv_path: str
    pdf_text_dict: dict
    analysis_report: str
    generated_code: str
    execution_result: str
    error_messages: list
    attempt_count: int
    max_attempts: int
    success: bool
    validation_score: float
    parsed_df: pd.DataFrame
    expected_df: pd.DataFrame

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
# NODE 1: PDF PARSER
# ============================================================================
def parse_pdf_node(state: AgentState) -> AgentState:
    """Node 1: Parse PDF and extract text"""
    print("\n" + "="*70)
    print("📄 NODE 1: PDF PARSING")
    print("="*70)
    logger.info("NODE 1: Starting PDF parsing")
    
    pdf_path = state["pdf_path"]
    pdf_text_dict = {}
    
    try:
        pdf_reader = PdfReader(pdf_path)
        number_of_pages = len(pdf_reader.pages)

        for i in range(number_of_pages):
            page = pdf_reader.pages[i]
            text = page.extract_text()
            if text:
                pdf_text_dict[i + 1] = text.strip()

        logger.info(f"✅ Successfully parsed {number_of_pages} pages from PDF")
        print(f"✅ Successfully parsed {number_of_pages} pages")
        
        state["pdf_text_dict"] = pdf_text_dict
        
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        print(f"❌ Error reading PDF: {e}")
        state["success"] = False
        if "error_messages" not in state:
            state["error_messages"] = []
        state["error_messages"].append(f"PDF parsing error: {e}")
    
    return state

# ============================================================================
# NODE 2: ANALYSER AGENT
# ============================================================================
def analyze_structure_node(state: AgentState) -> AgentState:
    """Node 2: Analyze PDF structure and patterns"""
    print("\n" + "="*70)
    print("🔍 NODE 2: STRUCTURE ANALYSIS")
    print("="*70)
    logger.info("NODE 2: Starting structure analysis")
    
    pdf_text_dict = state["pdf_text_dict"]
    
    # Initialize Gemini LLM
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # Build page samples
    sample_texts = []
    for page, text in pdf_text_dict.items():
        n = len(text)
        samples = (
            f"Page {page}:\nTOP:\n{text[:300]}\n"
            f"MIDDLE:\n{text[n//2:n//2+100]}\n"
            f"BOTTOM:\n{text[-200:]}\n"
        )
        sample_texts.append(samples)
    page_samples = "\n\n".join(sample_texts)
    
    # Analysis prompt
    analysis_prompt = PromptTemplate(
        input_variables=["page_samples"],
        template=(
            "You are an expert in **PDF document structure analysis**, **financial data extraction**, and **Python-based text parsing**.\n\n"
            "You are provided with raw text snippets extracted from a **bank statement PDF**.\n"
            "Each snippet includes portions from the top, middle, and bottom of several pages:\n\n"
            "{page_samples}\n\n"
            "Treat all data as **plain text**, not numeric. Do not calculate or summarize values. "
            "Words like 'Cr', 'Dr', 'Ltd', 'Balance', or similar terms are part of the document's structure.\n\n"
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
    
    print("🔄 Analyzing PDF structure...")
    analysis_chain = analysis_prompt | gemini_llm
    response = analysis_chain.invoke({"page_samples": page_samples})
    
    state["analysis_report"] = response.content
    logger.info("✅ PDF structural analysis complete")
    print("✅ Analysis complete")
    
    return state

# ============================================================================
# NODE 3: CODE GENERATOR AGENT
# ============================================================================
def generate_code_node(state: AgentState) -> AgentState:
    """Node 3: Generate parsing code based on analysis"""
    print("\n" + "="*70)
    print("🤖 NODE 3: CODE GENERATION")
    print("="*70)
    logger.info("NODE 3: Starting code generation")
    
    # Initialize Gemini LLM
    # g_llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     temperature=0,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    # )
    # Initialize Groq LLM
    g_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    
    # Load expected CSV for schema reference
    csv_path = state.get("csv_path", "")
    schema_info = ""
    if csv_path and os.path.exists(csv_path):
        try:
            expected_df = pd.read_csv(csv_path)
            schema_info = f"\n\n**Expected CSV Schema:**\nColumns: {list(expected_df.columns)}\nSample rows:\n{expected_df.head(3).to_string()}\n"
            state["expected_df"] = expected_df
        except Exception as e:
            logger.warning(f"Could not load expected CSV: {e}")
    
    code_prompt = PromptTemplate(
        input_variables=["analysis", "pdf_path", "schema_info"],
        template=(
            "You are an expert **Python developer for PDF-to-CSV conversion**.\n\n"
            "Below is a detailed **structural analysis** of a bank statement PDF:\n\n"
            "{analysis}\n\n"
            "{schema_info}\n\n"
            "Using this analysis, generate a **runnable Python script** that:\n"
            "1. Defines a function `parse(pdf_path: str) -> pd.DataFrame` that:\n"
            "   - Accepts a PDF file path as input\n"
            "   - Extracts text using **PyPDF2**\n"
            "   - Parses transactions based on detected table patterns\n"
            "   - Returns a pandas DataFrame with the transaction data\n"
            "2. The main block should:\n"
            "   - Check if PDF_PATH env var is set, else use '{pdf_path}'\n"
            "   - Call parse() with the PDF path\n"
            "   - Save the result to 'output_transactions.csv'\n"
            "   - Print 'Successfully parsed X transactions'\n"
            "   - Print 'Saved to output_transactions.csv'\n"
            "3. Match the expected CSV schema if provided\n"
            "4. Handle edge cases (missing data, split lines, etc.)\n\n"
            "### Critical Requirements\n"
            "- Must have a `parse(pdf_path: str) -> pd.DataFrame` function\n"
            "- Must import all necessary libraries (pandas, PyPDF2, re, etc.)\n"
            "- Must be directly runnable (no placeholders)\n"
            "- Focus on robust parsing with error handling\n"
            "- Include clear print statements for success\n\n"
            "Return only the **complete Python script** — no explanations or text outside the code block."
        )
    )
    
    print("🔄 Generating parsing code using Gemini...")
    code_chain = code_prompt | g_llm
    response = code_chain.invoke({
        "analysis": state["analysis_report"],
        "pdf_path": state["pdf_path"],
        "schema_info": schema_info
    })
    
    # Extract code from markdown
    code = extract_python_code(response.content)
    state["generated_code"] = code
    
    logger.info("✅ Code generation complete")
    print("✅ Code generated")
    
    return state

# ============================================================================
# NODE 4: EXECUTION & DEBUG AGENT
# ============================================================================
def execute_and_debug_node(state: AgentState) -> AgentState:
    """Node 4: Execute code and debug if necessary"""
    print("\n" + "="*70)
    print("🔧 NODE 4: EXECUTION & DEBUGGING")
    print("="*70)
    logger.info("NODE 4: Starting execution and debugging")
    
    attempt = state.get("attempt_count", 0)
    max_attempts = state.get("max_attempts", 3)
    
    if attempt >= max_attempts:
        logger.error("Max attempts reached")
        print(f"❌ Max attempts ({max_attempts}) reached")
        state["success"] = False
        return state
    
    state["attempt_count"] = attempt + 1
    current_attempt = state["attempt_count"]
    
    print(f"🔄 Attempt {current_attempt}/{max_attempts}")
    logger.info(f"Execution attempt {current_attempt}/{max_attempts}")
    
    # Write code to parser file
    bank_name = state["bank_name"]
    parser_path = Path(f"custom_parsers/{bank_name}_parser.py")
    parser_path.parent.mkdir(parents=True, exist_ok=True)
    parser_path.write_text(state["generated_code"])
    logger.info(f"Script written to {parser_path}")
    
    try:
        # Execute the parser
        print("▶️  Running script...")
        result = subprocess.run(
            [sys.executable, str(parser_path)],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PDF_PATH": state["pdf_path"]}
        )
        
        if result.returncode == 0:
            logger.info(f"✅ SUCCESS on attempt {current_attempt}")
            logger.info(f"Output:\n{result.stdout}")
            print("✅ Script executed successfully!")
            print("\n--- OUTPUT ---")
            print(result.stdout)
            
            state["execution_result"] = result.stdout
            state["success"] = True
            
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
        state["error_messages"].append(error_msg)
        state["success"] = False
        
        if current_attempt < max_attempts:
            state = invoke_debug_agent(state, error_msg)
        
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else e.stdout
        logger.error(f"Exit code {e.returncode}")
        logger.error(f"Error:\n{error_output}")
        print(f"❌ Execution failed (exit code {e.returncode})")
        print("\n--- ERROR ---")
        print(error_output)
        
        state["error_messages"].append(error_output)
        state["success"] = False
        
        # Try to auto-install missing packages
        if "ModuleNotFoundError" in error_output or "ImportError" in error_output:
            if install_missing_package(error_output):
                print("✓ Package installed. Retrying...\n")
                return state
        
        # Invoke debug agent if not at max attempts
        if current_attempt < max_attempts:
            print("\n🤖 Invoking Debug Agent...")
            state = invoke_debug_agent(state, error_output)
    
    return state

# ============================================================================
# NODE 5: VALIDATION AGENT
# ============================================================================
def validate_output_node(state: AgentState) -> AgentState:
    """Node 5: Validate parsed output against expected CSV"""
    print("\n" + "="*70)
    print("✅ NODE 5: VALIDATION")
    print("="*70)
    logger.info("NODE 5: Starting validation")
    
    if not state.get("success", False):
        print("⚠️ Skipping validation - execution not successful")
        return state
    
    # Check if output CSV was created
    output_csv = Path("output_transactions.csv")
    if not output_csv.exists():
        logger.error("Output CSV not found")
        print("❌ output_transactions.csv not found")
        state["validation_score"] = 0.0
        return state
    
    try:
        # Load parsed output
        parsed_df = pd.read_csv(output_csv)
        state["parsed_df"] = parsed_df
        
        print(f"📊 Parsed {len(parsed_df)} rows")
        logger.info(f"Parsed {len(parsed_df)} rows")
        
        # If expected CSV provided, compare
        if "expected_df" in state and state["expected_df"] is not None:
            expected_df = state["expected_df"]
            
            # Basic comparison
            print(f"📋 Expected {len(expected_df)} rows")
            
            # Calculate matching percentage
            matching_score = calculate_matching_score(parsed_df, expected_df)
            state["validation_score"] = matching_score
            
            print(f"\n🎯 Matching Score: {matching_score:.2f}%")
            logger.info(f"Validation score: {matching_score:.2f}%")
            
            # Detailed comparison
            if parsed_df.equals(expected_df):
                print("✅ DataFrames are identical!")
                logger.info("DataFrames are identical")
            else:
                print("⚠️ DataFrames differ - see details below:")
                compare_dataframes(parsed_df, expected_df)
        else:
            print("ℹ️ No expected CSV provided - skipping comparison")
            state["validation_score"] = 100.0
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        print(f"❌ Validation error: {e}")
        state["validation_score"] = 0.0
    
    return state

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_python_code(response_text):
    """Extract Python code from markdown code blocks"""
    pattern = r'```python\s*(.*?)```'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()

def install_missing_package(error_output):
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

def invoke_debug_agent(state: AgentState, error_output: str) -> AgentState:
    """Invoke Gemini to fix the code"""
    print("🤖 Invoking Debug Agent...")
    logger.info(f"Invoking debug agent (attempt {state['attempt_count']})...")
    
    # g_llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     temperature=0,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    # )
    g_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    debug_prompt = f"""You are an expert Python debugger. Fix the following script that failed with an error.

**Current Script:**
```python
{state['generated_code']}
```

**Error Message:**
{error_output}

**Instructions:**
1. Analyze the error carefully
2. Fix all syntax errors, undefined variables, missing imports, logic errors, etc.
3. Ensure the `parse(pdf_path: str) -> pd.DataFrame` function exists and works correctly
4. Ensure the main block reads PDF_PATH from env or uses a default
5. Ensure it saves to 'output_transactions.csv' and prints success messages
6. Return ONLY the complete fixed Python script in a ```python code block

**Important:** Return ONLY the code block, no explanations outside it."""
    
    response = g_llm.invoke(debug_prompt)
    fixed_code = extract_python_code(response.content)
    
    state["generated_code"] = fixed_code
    logger.info(f"Debug agent response received")
    print("✓ Debug Agent fixed the script. Retrying...\n")
    
    return state

def calculate_matching_score(parsed_df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
    """Calculate percentage of matching data between two DataFrames"""
    if parsed_df.empty and expected_df.empty:
        return 100.0
    
    if parsed_df.empty or expected_df.empty:
        return 0.0
    
    # Normalize column names
    parsed_df_norm = parsed_df.copy()
    expected_df_norm = expected_df.copy()
    parsed_df_norm.columns = parsed_df_norm.columns.str.strip().str.lower()
    expected_df_norm.columns = expected_df_norm.columns.str.strip().str.lower()
    
    # Check column overlap
    common_cols = list(set(parsed_df_norm.columns) & set(expected_df_norm.columns))
    if not common_cols:
        return 0.0
    
    # Compare rows
    total_cells = len(expected_df_norm) * len(common_cols)
    matching_cells = 0
    
    for col in common_cols:
        for idx in expected_df_norm.index:
            if idx in parsed_df_norm.index:
                expected_val = str(expected_df_norm.loc[idx, col]).strip()
                parsed_val = str(parsed_df_norm.loc[idx, col]).strip()
                if expected_val == parsed_val:
                    matching_cells += 1
    
    return (matching_cells / total_cells) * 100 if total_cells > 0 else 0.0

def compare_dataframes(parsed_df: pd.DataFrame, expected_df: pd.DataFrame):
    """Print detailed comparison between two DataFrames"""
    print("\n--- SHAPE COMPARISON ---")
    print(f"Parsed:   {parsed_df.shape}")
    print(f"Expected: {expected_df.shape}")
    
    print("\n--- COLUMN COMPARISON ---")
    parsed_cols = set(parsed_df.columns)
    expected_cols = set(expected_df.columns)
    print(f"Common:  {parsed_cols & expected_cols}")
    print(f"Missing: {expected_cols - parsed_cols}")
    print(f"Extra:   {parsed_cols - expected_cols}")

# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================
def should_retry(state: AgentState) -> str:
    """Decide whether to retry execution or move to validation"""
    if state.get("success", False):
        return "validate"
    elif state.get("attempt_count", 0) < state.get("max_attempts", 3):
        return "retry"
    else:
        return "end"

# ============================================================================
# LANGGRAPH WORKFLOW BUILDER
# ============================================================================
def build_agent_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("parse_pdf", parse_pdf_node)
    workflow.add_node("analyze", analyze_structure_node)
    workflow.add_node("generate", generate_code_node)
    workflow.add_node("execute", execute_and_debug_node)
    workflow.add_node("validate", validate_output_node)
    
    # Define edges
    workflow.set_entry_point("parse_pdf")
    workflow.add_edge("parse_pdf", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", "execute")
    
    # Conditional routing after execution
    workflow.add_conditional_edges(
        "execute",
        should_retry,
        {
            "validate": "validate",
            "retry": "execute",
            "end": END
        }
    )
    
    workflow.add_edge("validate", END)
    
    return workflow.compile()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Custom Bank Statement Parser - Multi-Agent System with LangGraph"
    )
    parser.add_argument("--target", required=True, help="Name of the bank (e.g., sbi, icici, hdfc)")
    parser.add_argument("--statement", help="Path to a sample statement PDF file")
    parser.add_argument("--expected-csv", help="Path to expected CSV output for validation")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum debug attempts (default: 3)")
    
    args = parser.parse_args()
    
    # Setup API keys
    setup_api_keys()
    
    # Resolve paths
    bank_name = args.target.lower()
    pdf_path = args.statement
    csv_path = args.expected_csv
    
    # Auto-discover files if not provided
    if not pdf_path:
        data_folder = Path(f"data/{bank_name}")
        if data_folder.exists():
            pdf_files = list(data_folder.glob("*.pdf"))
            if pdf_files:
                pdf_path = str(pdf_files[0])
                print(f"📄 Using PDF: {pdf_path}")
            
            if not csv_path:
                csv_files = list(data_folder.glob("*.csv"))
                if csv_files:
                    csv_path = str(csv_files[0])
                    print(f"📋 Using CSV: {csv_path}")
    
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        sys.exit(1)
    
    # Initialize state
    initial_state = {
        "pdf_path": pdf_path,
        "bank_name": bank_name,
        "csv_path": csv_path or "",
        "pdf_text_dict": {},
        "analysis_report": "",
        "generated_code": "",
        "execution_result": "",
        "error_messages": [],
        "attempt_count": 0,
        "max_attempts": args.max_attempts,
        "success": False,
        "validation_score": 0.0,
        "parsed_df": pd.DataFrame(),
        "expected_df": pd.DataFrame()
    }
    
    # Build and run the graph
    print("\n" + "="*70)
    print("🚀 CUSTOM BANK PARSER - LANGGRAPH MULTI-AGENT SYSTEM")
    print("="*70)
    logger.info("="*70)
    logger.info("Starting LangGraph Multi-Agent System")
    logger.info("="*70)
    
    app = build_agent_graph()
    final_state = app.invoke(initial_state)
    
    # Print summary
    print("\n" + "="*70)
    if final_state.get("success", False):
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"📊 Output saved to: output_transactions.csv")
        print(f"🎯 Validation Score: {final_state.get('validation_score', 0):.2f}%")
        print(f"📝 Parser saved to: custom_parsers/{bank_name}_parser.py")
    else:
        print("❌ PIPELINE FAILED")
        print("="*70)
        print(f"❌ Failed after {final_state.get('attempt_count', 0)} attempts")
        print(f"📋 Check bank_parser_agent.log for details")
    print("="*70)
    
    sys.exit(0 if final_state.get("success", False) else 1)

if __name__ == "__main__":
    main()