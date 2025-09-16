# Default configurations
PYTHON := python3
API_URL := http://localhost:8000
RESULTS_DIR := results
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

# Create necessary directories
setup:
	@echo "🔧 Setting up model evaluation environment..."
	@mkdir -p $(RESULTS_DIR)
	@$(PYTHON) -m pip install -r requirements.txt
	@echo "✅ Setup completed!"
	@echo "📁 Make sure ../test_dataset/ contains:"
	@echo "   • data.csv"
	@echo "   • images (*.jpg, *.png)"

# Check if API is running
check-api:
	@echo "🔍 Checking API status..."
	@curl -s $(API_URL)/ram > /dev/null && echo "✅ API is responding" || echo "❌ API is not accessible"

# Check if dataset exists
check-dataset:
	@echo "🔍 Checking dataset..."
	@if [ -d "../test_dataset" ]; then \
		echo "✅ test_dataset directory found"; \
		if [ -f "../test_dataset/data.csv" ]; then \
			echo "✅ data.csv file found"; \
			echo "📊 Dataset has $$(wc -l < ../test_dataset/data.csv) rows"; \
		else \
			echo "❌ data.csv file not found"; \
			exit 1; \
		fi; \
	else \
		echo "❌ test_dataset directory not found"; \
		exit 1; \
	fi

# Execute complete evaluation
evaluate: check-api check-dataset
	@echo "🚀 Starting model evaluation..."
	@$(PYTHON) evaluate_models.py \
		--api-url $(API_URL) \
		--output $(RESULTS_DIR)/evaluation_$(TIMESTAMP).json

# Execute evaluation without saving file
test-quick: check-api check-dataset
	@echo "⚡ Quick test (without saving file)..."
	@$(PYTHON) evaluate_models.py --api-url $(API_URL)

# Execute evaluation with custom output
evaluate-custom: check-dataset
	@echo "🚀 Starting custom evaluation..."
	@read -p "API URL [$(API_URL)]: " api_url; \
	api_url=$${api_url:-$(API_URL)}; \
	read -p "Output file name (without extension): " output_name; \
	$(PYTHON) evaluate_models.py \
		--api-url $$api_url \
		--output $(RESULTS_DIR)/$$output_name.json

# Generate HTML report from a JSON file
report:
	@echo "📄 Generating HTML report..."
	@if [ -z "$(JSON)" ]; then \
		echo "❌ Specify JSON file: make report JSON=results/evaluation_xxx.json"; \
		exit 1; \
	fi
	@$(PYTHON) generate_report.py $(JSON)

# List available results
list-results:
	@echo "📋 Available results:"
	@ls -la $(RESULTS_DIR)/*.json 2>/dev/null || echo "❌ No results found"

# Clean old results
clean:
	@echo "🧹 Cleaning old results..."
	@rm -rf $(RESULTS_DIR)/*.json $(RESULTS_DIR)/*.html 2>/dev/null || true
	@echo "✅ Cleanup completed!"

# Show environment status
status: check-api check-dataset
	@echo ""
	@echo "🎯 Test environment status:"
	@echo "  • API: ✅ Working"
	@echo "  • Dataset: ✅ Found"
	@echo "  • Saved results: $$(ls $(RESULTS_DIR)/*.json 2>/dev/null | wc -l) files"

# Show help
help:
	@echo "📋 Available commands for model testing:"
	@echo ""
	@echo "  setup              - Setup environment (install dependencies)"
	@echo "  check-api          - Check if API is running"
	@echo "  check-dataset      - Check if dataset is present"
	@echo "  evaluate           - Execute complete evaluation and save result"
	@echo "  test-quick         - Quick test without saving file"
	@echo "  evaluate-custom    - Execute evaluation with custom parameters"
	@echo "  report JSON=file   - Generate HTML report from JSON file"
	@echo "  list-results       - List available results"
	@echo "  clean              - Remove old results"
	@echo "  status             - Show general environment status"
	@echo "  help               - Show this help"
	@echo ""
	@echo "📁 Expected structure:"
	@echo "  ../test_dataset/data.csv     - CSV file with test data"
	@echo "  ../test_dataset/*.jpg        - Test images"
	@echo "  ./results/                   - Folder where results are saved"
	@echo ""
	@echo "💡 Examples:"
	@echo "  make evaluate                           # Complete evaluation"
	@echo "  make report JSON=results/eval_xxx.json  # Generate report"

# Configure environment variables
.PHONY: setup check-api check-dataset evaluate test-quick evaluate-custom report list-results clean status help

# By default, show help
.DEFAULT_GOAL := help