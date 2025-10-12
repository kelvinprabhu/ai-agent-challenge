# Custom Bank Statement Parser - Multi-Agent System

An intelligent AI agent system that automatically generates custom parsers for bank statement PDFs using LangGraph orchestration and LLM-powered code generation.

## 🎯 Overview

This system uses a multi-agent architecture to:
1. Parse bank statement PDFs
2. Analyze document structure
3. Generate custom parsing code
4. Self-debug and fix errors
5. Validate output against expected results

## 🏗️ Architecture

### LangGraph Node Design

```
┌──────────────────────────────────────────────────────────────┐
│                    LANGGRAPH WORKFLOW                         │
└──────────────────────────────────────────────────────────────┘

    ┌─────────────┐
    │   START     │
    └──────┬──────┘
           │
           v
    ┌─────────────────────────────────────────────────┐
    │  NODE 1: PDF PARSER                             │
    │  • Extracts text from PDF using PyPDF2          │
    │  • Stores page-wise text in dictionary          │
    │  • Handles multi-page documents                 │
    └──────────────────┬──────────────────────────────┘
                       │
                       v
    ┌─────────────────────────────────────────────────┐
    │  NODE 2: STRUCTURE ANALYZER                     │
    │  • Analyzes PDF layout patterns                 │
    │  • Identifies table structure                   │
    │  • Detects column headers & delimiters          │
    │  • Uses Gemini 2.0 Flash for analysis           │
    └──────────────────┬──────────────────────────────┘
                       │
                       v
    ┌─────────────────────────────────────────────────┐
    │  NODE 3: CODE GENERATOR                         │
    │  • Generates parse() function                   │
    │  • Creates runnable Python script               │
    │  • Matches expected CSV schema                  │
    │  • Uses Gemini 2.0 Flash for generation         │
    └──────────────────┬──────────────────────────────┘
                       │
                       v
    ┌─────────────────────────────────────────────────┐
    │  NODE 4: EXECUTOR & DEBUG AGENT                 │
    │  • Runs generated parser code                   │
    │  • Catches execution errors                     │
    │  • Auto-installs missing packages               │
    │  • Invokes LLM for code fixes (max 3 attempts)  │
    └──────────────────┬──────────────────────────────┘
                       │
                       v
                 ┌─────────┐
                 │ Success?│
                 └────┬────┘
                      │
            ┌─────────┴─────────┐
            │                   │
          YES                  NO
            │                   │
            v                   v
    ┌───────────────┐    ┌──────────────┐
    │  Attempts < 3?│    │   VALIDATE   │
    │     Retry     │    │              │
    └───────┬───────┘    └──────┬───────┘
            │                   │
            └─────────┬─────────┘
                      │
                      v
              ┌───────────────┐
              │   NODE 5:     │
              │  VALIDATION   │
              │  • Loads CSV  │
              │  • Compares   │
              │  • Scores %   │
              └───────┬───────┘
                      │
                      v
                  ┌───────┐
                  │  END  │
                  └───────┘
```

### State Flow

The system maintains a shared `AgentState` that flows through all nodes:

```python
AgentState:
  ├── pdf_path: str              # Input PDF path
  ├── bank_name: str             # Target bank identifier
  ├── csv_path: str              # Expected CSV path
  ├── pdf_text_dict: dict        # Parsed PDF text by page
  ├── analysis_report: str       # Structure analysis results
  ├── generated_code: str        # Generated parser code
  ├── execution_result: str      # Execution output
  ├── error_messages: list       # Error history
  ├── attempt_count: int         # Current attempt number
  ├── max_attempts: int          # Maximum retry limit
  ├── success: bool              # Execution status
  ├── validation_score: float    # Matching percentage
  ├── parsed_df: DataFrame       # Parsed output data
  └── expected_df: DataFrame     # Expected output data