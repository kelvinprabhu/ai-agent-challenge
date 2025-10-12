#!/bin/bash

# Quick Start Script for Bank Statement Parser Agent
# Usage: ./quickstart.sh [bank_name]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "========================================================================"
echo "  🏦 Bank Statement Parser - Quick Start Setup"
echo "========================================================================"
echo -e "${NC}"

# Check Python version
echo -e "${BLUE}📋 Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✅ Found Python $PYTHON_VERSION${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "\n${BLUE}🔧 Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "\n${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${BLUE}🔌 Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${BLUE}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✅ pip upgraded${NC}"

# Install dependencies
echo -e "\n${BLUE}📦 Installing dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt not found, installing core packages...${NC}"
    pip install langchain langchain-groq langchain-google-genai langgraph PyPDF2 pandas python-dotenv --quiet
    echo -e "${GREEN}✅ Core packages installed${NC}"
fi

# Check for .env file
echo -e "\n${BLUE}🔑 Checking API keys...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating template...${NC}"
    cat > .env << EOF
# API Keys for Bank Statement Parser
# Get your keys from:
# - Groq: https://console.groq.com/keys
# - Google Gemini: https://makersuite.google.com/app/apikey

GROQ_API_KEY=your-groq-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
EOF
    echo -e "${YELLOW}📝 Please edit .env file and add your API keys${NC}"
    echo -e "${YELLOW}   Then run this script again${NC}"
    exit 0
fi

# Validate API keys
source .env
if [ "$GROQ_API_KEY" = "your-groq-api-key-here" ] || [ -z "$GROQ_API_KEY" ]; then
    echo -e "${RED}❌ GROQ_API_KEY not set in .env file${NC}"
    echo -e "${YELLOW}   Please edit .env and add your Groq API key${NC}"
    exit 1
fi

if [ "$GOOGLE_API_KEY" = "your-google-api-key-here" ] || [ -z "$GOOGLE_API_KEY" ]; then
    echo -e "${RED}❌ GOOGLE_API_KEY not set in .env file${NC}"
    echo -e "${YELLOW}   Please edit .env and add your Google API key${NC}"
    exit 1
fi

echo -e "${GREEN}✅ API keys configured${NC}"

# Create necessary directories
echo -e "\n${BLUE}📁 Creating directories...${NC}"
mkdir -p data/icici data/sbi data/hdfc custom_parsers
echo -e "${GREEN}✅ Directories created${NC}"

# Check if bank name provided
BANK_NAME=${1:-}

if [ -z "$BANK_NAME" ]; then
    echo -e "\n${BLUE}========================================================================"
    echo "  ✅ Setup Complete!"
    echo "========================================================================${NC}"
    echo -e "\n${GREEN}Next steps:${NC}"
    echo "1. Add your bank statement PDF to: data/<bank_name>/<bank_name>_sample.pdf"
    echo "2. Add expected CSV to: data/<bank_name>/<bank_name>_expected.csv"
    echo "3. Run the agent:"
    echo -e "   ${BLUE}python agent.py --target <bank_name>${NC}"
    echo ""
    echo "Example:"
    echo -e "   ${BLUE}python agent.py --target icici${NC}"
    echo ""
    echo "Or run this script with a bank name:"
    echo -e "   ${BLUE}./quickstart.sh icici${NC}"
    exit 0
fi

# Run agent for specified bank
BANK_LOWER=$(echo "$BANK_NAME" | tr '[:upper:]' '[:lower:]')

echo -e "\n${BLUE}========================================================================"
echo "  🚀 Running Agent for: ${BANK_LOWER}"
echo "========================================================================${NC}"

# Check if PDF exists
PDF_PATH="data/${BANK_LOWER}/${BANK_LOWER}_sample.pdf"
if [ ! -f "$PDF_PATH" ]; then
    echo -e "${RED}❌ PDF not found: $PDF_PATH${NC}"
    echo -e "${YELLOW}   Please add your bank statement PDF to this location${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found PDF: $PDF_PATH${NC}"

# Check if expected CSV exists (optional)
CSV_PATH="data/${BANK_LOWER}/${BANK_LOWER}_expected.csv"
if [ ! -f "$CSV_PATH" ]; then
    echo -e "${YELLOW}⚠️  Expected CSV not found: $CSV_PATH${NC}"
    echo -e "${YELLOW}   Validation will be skipped${NC}"
    CSV_ARG=""
else
    echo -e "${GREEN}✅ Found expected CSV: $CSV_PATH${NC}"
    CSV_ARG="--expected-csv $CSV_PATH"
fi

# Run the agent
echo -e "\n${BLUE}🤖 Starting Agent...${NC}\n"
python agent.py --target "$BANK_LOWER" --statement "$PDF_PATH" $CSV_ARG

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================================================"
    echo "  ✅ Agent Execution Successful!"
    echo "========================================================================${NC}"
    echo ""
    echo "Generated files:"
    echo -e "  • Parser: ${BLUE}custom_parsers/${BANK_LOWER}_parser.py${NC}"
    echo -e "  • Output: ${BLUE}output_transactions.csv${NC}"
    echo -e "  • Logs:   ${BLUE}bank_parser_agent.log${NC}"
    echo ""
    echo "Test your parser:"
    echo -e "  ${BLUE}python test_parser.py --bank $BANK_LOWER${NC}"
else
    echo -e "\n${RED}========================================================================"
    echo "  ❌ Agent Execution Failed"
    echo "========================================================================${NC}"
    echo ""
    echo "Check the logs for details:"
    echo -e "  ${BLUE}tail -50 bank_parser_agent.log${NC}"
    exit 1
fi