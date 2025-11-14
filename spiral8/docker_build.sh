#!/usr/bin/env bash
# Grand Unified Theory - Docker Build and Test Script
# ====================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}GUT Docker Build & Test${NC}"
echo -e "${BLUE}================================${NC}"

# Check Docker availability
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    echo "  Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker available${NC}"

# Check Docker Compose availability
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${YELLOW}⚠ Docker Compose not found (optional)${NC}"
    COMPOSE_AVAILABLE=false
else
    echo -e "${GREEN}✓ Docker Compose available${NC}"
    COMPOSE_AVAILABLE=true
fi

# Build image
echo ""
echo -e "${BLUE}Building Docker image...${NC}"
docker build -t gut-testing:latest . || {
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ Image built successfully${NC}"

# Create output directories
echo ""
echo -e "${BLUE}Creating output directories...${NC}"
mkdir -p output plots logs ligo_data
echo -e "${GREEN}✓ Directories created${NC}"

# Run tests
echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Running Tests${NC}"
echo -e "${BLUE}================================${NC}"

echo ""
echo -e "${YELLOW}Test 1: Main Validation${NC}"
docker run --rm \
    -v "$(pwd)/output:/gut/output" \
    gut-testing:latest \
    python grand_unified_theory.py

echo ""
echo -e "${YELLOW}Test 2: Interactive Demo${NC}"
docker run --rm \
    -v "$(pwd)/output:/gut/output" \
    -v "$(pwd)/plots:/gut/plots" \
    gut-testing:latest \
    python gut_demo.py

echo ""
echo -e "${YELLOW}Test 3: C Precision Engine${NC}"
docker run --rm \
    -v "$(pwd)/output:/gut/output" \
    gut-testing:latest \
    gut_engine validate-all

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}All Tests Complete!${NC}"
echo -e "${GREEN}================================${NC}"

echo ""
echo "Output files:"
echo "  - output/gut_report.json"
echo "  - plots/*.png"
echo "  - logs/*.log"

echo ""
echo "Next steps:"
echo "  1. Download LIGO data:"
echo "     docker run --rm -it -v \$(pwd)/ligo_data:/gut/ligo_data gut-testing python download_data.py"
echo ""
echo "  2. Run full analysis:"
echo "     docker run --rm -v \$(pwd)/output:/gut/output gut-testing python gut_data_analysis.py"
echo ""
echo "  3. Interactive shell:"
echo "     docker run --rm -it gut-testing /bin/bash"
echo ""
if [ "$COMPOSE_AVAILABLE" = true ]; then
    echo "  4. Run all services with Docker Compose:"
    echo "     docker-compose up"
fi
