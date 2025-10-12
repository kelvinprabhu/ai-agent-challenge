"""
Test script to validate generated parsers against expected CSV output.
Usage: python test_parser.py --bank icici
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import importlib.util

def load_parser_module(bank_name: str):
    """Dynamically load the generated parser module"""
    parser_path = Path(f"custom_parsers/{bank_name}_parser.py")
    
    if not parser_path.exists():
        print(f"❌ Parser not found: {parser_path}")
        print(f"💡 Run: python agent.py --target {bank_name}")
        return None
    
    spec = importlib.util.spec_from_file_location(f"{bank_name}_parser", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module

def calculate_matching_percentage(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    """Calculate detailed matching statistics between two DataFrames"""
    results = {
        "exact_match": False,
        "shape_match": False,
        "columns_match": False,
        "row_count_match": False,
        "cell_match_percentage": 0.0,
        "matching_cells": 0,
        "total_cells": 0,
        "differences": []
    }
    
    # Check exact match
    results["exact_match"] = df1.equals(df2)
    
    # Check shape
    results["shape_match"] = df1.shape == df2.shape
    
    # Normalize column names
    df1_cols = set(df1.columns.str.strip().str.lower())
    df2_cols = set(df2.columns.str.strip().str.lower())
    
    results["columns_match"] = df1_cols == df2_cols
    results["row_count_match"] = len(df1) == len(df2)
    
    # Calculate cell-level matching
    if results["columns_match"] and results["row_count_match"]:
        common_cols = list(df1_cols & df2_cols)
        total_cells = len(df1) * len(common_cols)
        matching_cells = 0
        
        # Normalize column names for comparison
        df1_normalized = df1.copy()
        df2_normalized = df2.copy()
        df1_normalized.columns = df1_normalized.columns.str.strip().str.lower()
        df2_normalized.columns = df2_normalized.columns.str.strip().str.lower()
        
        for col in common_cols:
            for idx in df1_normalized.index:
                if idx < len(df2_normalized):
                    val1 = str(df1_normalized.loc[idx, col]).strip()
                    val2 = str(df2_normalized.loc[idx, col]).strip()
                    
                    if val1 == val2:
                        matching_cells += 1
                    else:
                        results["differences"].append({
                            "row": idx,
                            "column": col,
                            "parsed": val1,
                            "expected": val2
                        })
        
        results["matching_cells"] = matching_cells
        results["total_cells"] = total_cells
        results["cell_match_percentage"] = (matching_cells / total_cells * 100) if total_cells > 0 else 0.0
    
    return results

def print_comparison_report(parsed_df: pd.DataFrame, expected_df: pd.DataFrame, results: dict):
    """Print detailed comparison report"""
    print("\n" + "="*70)
    print("📊 VALIDATION REPORT")
    print("="*70)
    
    # Summary
    print("\n✅ SUMMARY")
    print(f"  • Exact Match: {'YES ✅' if results['exact_match'] else 'NO ❌'}")
    print(f"  • Cell Match: {results['cell_match_percentage']:.2f}%")
    print(f"  • Matching Cells: {results['matching_cells']}/{results['total_cells']}")
    
    # Shape comparison
    print("\n📐 SHAPE")
    print(f"  • Parsed:   {parsed_df.shape}")
    print(f"  • Expected: {expected_df.shape}")
    print(f"  • Match: {'YES ✅' if results['shape_match'] else 'NO ❌'}")
    
    # Column comparison
    print("\n📋 COLUMNS")
    parsed_cols = set(parsed_df.columns.str.strip().str.lower())
    expected_cols = set(expected_df.columns.str.strip().str.lower())
    
    print(f"  • Parsed:   {sorted(parsed_cols)}")
    print(f"  • Expected: {sorted(expected_cols)}")
    print(f"  • Common:   {sorted(parsed_cols & expected_cols)}")
    
    if expected_cols - parsed_cols:
        print(f"  • Missing:  {sorted(expected_cols - parsed_cols)} ❌")
    if parsed_cols - expected_cols:
        print(f"  • Extra:    {sorted(parsed_cols - expected_cols)} ⚠️")
    
    # Row count comparison
    print(f"\n📊 ROWS")
    print(f"  • Parsed:   {len(parsed_df)}")
    print(f"  • Expected: {len(expected_df)}")
    print(f"  • Match: {'YES ✅' if results['row_count_match'] else 'NO ❌'}")
    
    # Show sample differences (first 5)
    if results["differences"]:
        print(f"\n⚠️  DIFFERENCES (showing first 5 of {len(results['differences'])})")
        for i, diff in enumerate(results["differences"][:5], 1):
            print(f"\n  {i}. Row {diff['row']}, Column '{diff['column']}':")
            print(f"     Parsed:   '{diff['parsed']}'")
            print(f"     Expected: '{diff['expected']}'")
    
    # Data sample preview
    print(f"\n📄 SAMPLE DATA (first 3 rows)")
    print("\nParsed:")
    print(parsed_df.head(3).to_string())
    print("\nExpected:")
    print(expected_df.head(3).to_string())
    
    print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(
        description="Test generated bank statement parser"
    )
    parser.add_argument("--bank", required=True, help="Bank name (e.g., icici, sbi)")
    parser.add_argument("--pdf", help="Path to PDF file (default: data/<bank>/<bank>_sample.pdf)")
    parser.add_argument("--expected", help="Path to expected CSV (default: data/<bank>/<bank>_expected.csv)")
    parser.add_argument("--strict", action="store_true", help="Exit with error if not 100%% match")
    
    args = parser.parse_args()
    bank_name = args.bank.lower()
    
    # Resolve paths
    pdf_path = args.pdf or f"data/{bank_name}/{bank_name}_sample.pdf"
    expected_csv = args.expected or f"data/{bank_name}/{bank_name}_expected.csv"
    
    print("="*70)
    print(f"🧪 TESTING PARSER: {bank_name.upper()}")
    print("="*70)
    print(f"📄 PDF: {pdf_path}")
    print(f"📋 Expected CSV: {expected_csv}")
    print("="*70)
    
    # Validate files exist
    if not Path(pdf_path).exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not Path(expected_csv).exists():
        print(f"❌ Error: Expected CSV not found: {expected_csv}")
        sys.exit(1)
    
    # Load parser module
    print("\n📦 Loading parser module...")
    parser_module = load_parser_module(bank_name)
    
    if parser_module is None:
        sys.exit(1)
    
    if not hasattr(parser_module, 'parse'):
        print(f"❌ Error: Parser module missing 'parse()' function")
        sys.exit(1)
    
    print("✅ Parser module loaded successfully")
    
    # Execute parser
    print(f"\n🔄 Parsing PDF: {pdf_path}...")
    try:
        parsed_df = parser_module.parse(pdf_path)
        print(f"✅ Parsing complete - {len(parsed_df)} rows extracted")
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load expected CSV
    print(f"\n📥 Loading expected CSV...")
    try:
        expected_df = pd.read_csv(expected_csv)
        print(f"✅ Expected CSV loaded - {len(expected_df)} rows")
    except Exception as e:
        print(f"❌ Failed to load expected CSV: {e}")
        sys.exit(1)
    
    # Calculate matching
    print(f"\n🔍 Comparing outputs...")
    results = calculate_matching_percentage(parsed_df, expected_df)
    
    # Print detailed report
    print_comparison_report(parsed_df, expected_df, results)
    
    # Final verdict
    print("\n" + "="*70)
    if results["exact_match"]:
        print("🎉 TEST PASSED: Perfect match!")
        print("="*70)
        sys.exit(0)
    elif results["cell_match_percentage"] >= 95.0:
        print(f"✅ TEST PASSED: {results['cell_match_percentage']:.2f}% match (>95%)")
        print("="*70)
        sys.exit(0 if not args.strict else 1)
    elif results["cell_match_percentage"] >= 80.0:
        print(f"⚠️  TEST WARNING: {results['cell_match_percentage']:.2f}% match (80-95%)")
        print("💡 Consider reviewing differences above")
        print("="*70)
        sys.exit(1 if args.strict else 0)
    else:
        print(f"❌ TEST FAILED: {results['cell_match_percentage']:.2f}% match (<80%)")
        print("💡 Significant differences detected - review parser logic")
        print("="*70)
        sys.exit(1)

if __name__ == "__main__":
    main()