# Default configurations
PYTHON := python3
API_URL := http://localhost:8000
RESULTS_DIR := results
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

# Create necessary directories
setup:
	@echo "🔧 Setting up AI vs Dermatologists comparison environment..."
	@mkdir -p $(RESULTS_DIR)
	@$(PYTHON) -m pip install -r requirements.txt
	@echo "✅ Setup completed!"
	@echo "📁 Make sure ../test_dataset/ contains:"
	@echo "   • data.csv (with columns: image, binary, dx, clinical_impression_1, clinical_impression_2, clinical_impression_3)"
	@echo "   • images (*.jpg, *.png)"

# Check if API is running
check-api:
	@echo "🔍 Checking API status..."
	@curl -s $(API_URL)/ram > /dev/null && echo "✅ API is responding" || echo "❌ API is not accessible"

# Check if dataset exists and has necessary columns
check-dataset:
	@echo "🔍 Checking dataset..."
	@if [ -d "../test_dataset" ]; then \
		echo "✅ test_dataset directory found"; \
		if [ -f "../test_dataset/data.csv" ]; then \
			echo "✅ data.csv file found"; \
			echo "📊 Dataset has $$(wc -l < ../test_dataset/data.csv) rows"; \
			echo "🔍 Checking required columns..."; \
			$(PYTHON) -c "import pandas as pd; df=pd.read_csv('../test_dataset/data.csv'); required=['image','binary','dx','clinical_impression_1','clinical_impression_2','clinical_impression_3']; missing=[col for col in required if col not in df.columns]; print('✅ All required columns found' if not missing else f'❌ Missing columns: {missing}'); exit(1 if missing else 0)"; \
		else \
			echo "❌ data.csv file not found"; \
			exit 1; \
		fi; \
	else \
		echo "❌ test_dataset directory not found"; \
		exit 1; \
	fi

# Check if backend is running via Docker
check-backend:
	@echo "🐳 Checking if backend is running..."
	@if docker compose ps | grep -q "running"; then \
		echo "✅ Docker backend is running"; \
	else \
		echo "⚠️  Docker backend doesn't seem to be running"; \
		echo "💡 Run: docker compose up -d"; \
	fi

# Execute complete comparison
compare: check-api check-dataset
	@echo "🚀 Starting AI vs Dermatologists comparison..."
	@$(PYTHON) compare_ai_dermatologists.py \
		--api-url $(API_URL) \
		--output $(RESULTS_DIR)/comparison_$(TIMESTAMP).json
	@echo "✅ Comparison completed! Results saved in: $(RESULTS_DIR)/comparison_$(TIMESTAMP).json"

# Execute comparison without saving file
test-quick: check-api check-dataset
	@echo "⚡ Quick comparison test (without saving file)..."
	@$(PYTHON) compare_ai_dermatologists.py --api-url $(API_URL)

# Execute comparison with custom parameters
compare-custom: check-dataset
	@echo "🚀 Starting custom comparison..."
	@read -p "API URL [$(API_URL)]: " api_url; \
	api_url=$${api_url:-$(API_URL)}; \
	read -p "Output file name (without extension): " output_name; \
	$(PYTHON) compare_ai_dermatologists.py \
		--api-url $$api_url \
		--output $(RESULTS_DIR)/$$output_name.json

# Generate HTML report from a comparison JSON
report:
	@echo "📄 Generating HTML comparison report..."
	@if [ -z "$(JSON)" ]; then \
		echo "❌ Specify JSON file: make report JSON=results/comparison_xxx.json"; \
		exit 1; \
	fi
	@if [ ! -f "$(JSON)" ]; then \
		echo "❌ File not found: $(JSON)"; \
		exit 1; \
	fi
	@output_html=$$(echo "$(JSON)" | sed 's/\.json$$/_report.html/'); \
	$(PYTHON) generate_comparison_report.py --input $(JSON) --output $$output_html; \
	echo "📊 Report generated: $$output_html"

# Execute complete workflow: comparison + report
full-analysis: check-backend check-api check-dataset
	@echo "🎯 Running complete analysis (comparison + report)..."
	@output_json=$(RESULTS_DIR)/comparison_$(TIMESTAMP).json; \
	output_html=$(RESULTS_DIR)/comparison_$(TIMESTAMP)_report.html; \
	$(PYTHON) compare_ai_dermatologists.py \
		--api-url $(API_URL) \
		--output $$output_json && \
	$(PYTHON) generate_comparison_report.py \
		--input $$output_json \
		--output $$output_html && \
	echo "🎉 Complete analysis finished!" && \
	echo "📊 Data: $$output_json" && \
	echo "📄 Report: $$output_html" && \
	echo "💡 Open the HTML file in browser to view!"

# Generate report from latest result
report-latest:
	@echo "📄 Generating report from most recent result..."
	@latest_json=$$(ls -t $(RESULTS_DIR)/comparison_*.json 2>/dev/null | head -n1); \
	if [ -z "$$latest_json" ]; then \
		echo "❌ No comparison results found"; \
		echo "💡 Run: make compare"; \
		exit 1; \
	fi; \
	echo "📊 Using file: $$latest_json"; \
	output_html=$$(echo "$$latest_json" | sed 's/\.json$$/_report.html/'); \
	$(PYTHON) generate_comparison_report.py --input $$latest_json --output $$output_html; \
	echo "📄 Report generated: $$output_html"

# Open latest report in browser
open-report:
	@echo "🌐 Opening latest report in browser..."
	@latest_html=$$(ls -t $(RESULTS_DIR)/comparison_*_report.html 2>/dev/null | head -n1); \
	if [ -z "$$latest_html" ]; then \
		echo "❌ No reports found"; \
		echo "💡 Run: make report-latest"; \
		exit 1; \
	fi; \
	echo "📄 Opening: $$latest_html"; \
	if command -v open > /dev/null; then \
		open $$latest_html; \
	elif command -v xdg-open > /dev/null; then \
		xdg-open $$latest_html; \
	else \
		echo "🖥️  Manually open file: $$latest_html"; \
	fi

# List available comparison results
list-results:
	@echo "📋 Available comparison results:"
	@echo ""
	@echo "🔬 JSON files (raw data):"
	@ls -la $(RESULTS_DIR)/comparison_*.json 2>/dev/null | awk '{print "  📊 " $$9 " (" $$5 " bytes, " $$6 " " $$7 " " $$8 ")"}' || echo "  ❌ No JSON files found"
	@echo ""
	@echo "📄 HTML reports:"
	@ls -la $(RESULTS_DIR)/comparison_*_report.html 2>/dev/null | awk '{print "  🌐 " $$9 " (" $$5 " bytes, " $$6 " " $$7 " " $$8 ")"}' || echo "  ❌ No HTML reports found"

# Compare results between different executions
compare-results:
	@echo "📊 Comparing results between executions..."
	@json_files=$$(ls $(RESULTS_DIR)/comparison_*.json 2>/dev/null); \
	if [ -z "$$json_files" ]; then \
		echo "❌ No results found to compare"; \
		exit 1; \
	fi; \
	echo "📋 Available files:"; \
	for file in $$json_files; do \
		timestamp=$$(basename $$file .json | sed 's/comparison_//'); \
		echo "  📊 $$file ($$timestamp)"; \
	done; \
	echo "💡 Use 'make report JSON=file.json' to generate specific reports"

# Clean old results
clean:
	@echo "🧹 Cleaning comparison results..."
	@count_json=$$(ls $(RESULTS_DIR)/comparison_*.json 2>/dev/null | wc -l); \
	count_html=$$(ls $(RESULTS_DIR)/comparison_*_report.html 2>/dev/null | wc -l); \
	rm -f $(RESULTS_DIR)/comparison_*.json $(RESULTS_DIR)/comparison_*_report.html 2>/dev/null || true; \
	echo "✅ Cleanup completed!"; \
	echo "🗑️  Removed: $$count_json JSON(s) + $$count_html HTML(s)"

# Clean only old files (keep 3 most recent)
clean-old:
	@echo "🧹 Keeping only the 3 most recent results..."
	@ls -t $(RESULTS_DIR)/comparison_*.json 2>/dev/null | tail -n +4 | xargs rm -f 2>/dev/null || true
	@ls -t $(RESULTS_DIR)/comparison_*_report.html 2>/dev/null | tail -n +4 | xargs rm -f 2>/dev/null || true
	@echo "✅ Selective cleanup completed!"

# Start backend if not running
start-backend:
	@echo "🚀 Starting Docker backend..."
	@cd .. && docker compose up -d
	@echo "⏳ Waiting for backend to initialize..."
	@sleep 10
	@make check-api

# Stop backend
stop-backend:
	@echo "🛑 Stopping Docker backend..."
	@cd .. && docker compose down
	@echo "✅ Backend stopped!"

# Show complete environment status
status: check-backend check-api check-dataset
	@echo ""
	@echo "🎯 Comparison environment status:"
	@echo "  • Docker Backend: ✅ Running"
	@echo "  • API: ✅ Working"
	@echo "  • Dataset: ✅ Found and valid"
	@json_count=$$(ls $(RESULTS_DIR)/comparison_*.json 2>/dev/null | wc -l); \
	html_count=$$(ls $(RESULTS_DIR)/comparison_*_report.html 2>/dev/null | wc -l); \
	echo "  • Saved results: $$json_count JSON(s), $$html_count report(s)"

# Check Python dependencies
check-deps:
	@echo "🔍 Checking Python dependencies..."
	@$(PYTHON) -c "import requests, pandas, numpy, sklearn, matplotlib, seaborn, plotly; print('✅ All dependencies are installed')" 2>/dev/null || \
	(echo "❌ Some dependencies are missing"; echo "💡 Run: make setup")

# Show dataset statistics
dataset-stats:
	@echo "📊 Dataset statistics:"
	@$(PYTHON) -c "\
import pandas as pd; \
df = pd.read_csv('../test_dataset/data.csv'); \
print(f'📋 Total images: {len(df)}'); \
print(f'🔍 Binary classification:'); \
print(df['binary'].value_counts().to_string().replace('\n', '\n   ')); \
print(f'🎯 Specific diagnoses:'); \
print(df['dx'].value_counts().head(10).to_string().replace('\n', '\n   ')); \
print(f'👥 Non-null clinical impressions:'); \
for i in range(1, 4): \
    col = f'clinical_impression_{i}'; \
    valid = df[col].notna().sum(); \
    print(f'   Dermatologist {i}: {valid}/{len(df)} ({valid/len(df)*100:.1f}%)');"

# Run example analysis (first 10 images)
demo:
	@echo "🎮 Running demonstration with dataset subset..."
	@echo "⚠️  Creating temporary dataset with first 10 images..."
	@$(PYTHON) -c "\
import pandas as pd; \
df = pd.read_csv('../test_dataset/data.csv'); \
df_demo = df.head(10); \
df_demo.to_csv('../test_dataset/data_demo.csv', index=False); \
print('✅ Demo dataset created: ../test_dataset/data_demo.csv')"
	@mv ../test_dataset/data.csv ../test_dataset/data_full.csv
	@mv ../test_dataset/data_demo.csv ../test_dataset/data.csv
	@make compare
	@mv ../test_dataset/data.csv ../test_dataset/data_demo.csv
	@mv ../test_dataset/data_full.csv ../test_dataset/data.csv
	@echo "🎉 Demo completed! Original dataset restored."

# Show help
help:
	@echo "📋 Available commands for AI vs Dermatologists comparison:"
	@echo ""
	@echo "🔧 Setup and Verification:"
	@echo "  setup              - Setup environment (install dependencies)"
	@echo "  check-backend      - Check if Docker backend is running"
	@echo "  check-api          - Check if API is responding"
	@echo "  check-dataset      - Check dataset and required columns"
	@echo "  check-deps         - Check Python dependencies"
	@echo "  status             - Complete environment status"
	@echo ""
	@echo "🔬 Comparison Execution:"
	@echo "  compare            - Execute complete comparison"
	@echo "  test-quick         - Quick test without saving file"
	@echo "  compare-custom     - Comparison with custom parameters"
	@echo "  full-analysis      - Comparison + automatic report"
	@echo "  demo               - Demonstration with first 10 images"
	@echo ""
	@echo "📄 Reports:"
	@echo "  report JSON=file   - Generate HTML report from specific JSON file"
	@echo "  report-latest      - Generate report from most recent result"
	@echo "  open-report        - Open latest report in browser"
	@echo ""
	@echo "📊 Results Management:"
	@echo "  list-results       - List available results"
	@echo "  compare-results    - Compare results between executions"
	@echo "  clean              - Remove all results"
	@echo "  clean-old          - Remove old results (keep 3 most recent)"
	@echo ""
	@echo "🐳 Backend Control:"
	@echo "  start-backend      - Start Docker backend"
	@echo "  stop-backend       - Stop Docker backend"
	@echo ""
	@echo "📈 Analysis:"
	@echo "  dataset-stats      - Show dataset statistics"
	@echo ""
	@echo "💡 Recommended workflow for first execution:"
	@echo "  1. make setup                    # Setup environment"
	@echo "  2. make start-backend            # Start backend"
	@echo "  3. make full-analysis            # Complete comparison + report"
	@echo "  4. make open-report              # View results"
	@echo ""
	@echo "📁 Expected structure:"
	@echo "  ../test_dataset/data.csv         - CSV with required columns"
	@echo "  ../test_dataset/*.jpg            - Test images"
	@echo "  ./results/                       - Results and reports"

# Configure environment variables and special rules
.PHONY: setup check-api check-dataset check-backend check-deps compare test-quick compare-custom
.PHONY: report report-latest open-report full-analysis list-results compare-results
.PHONY: clean clean-old start-backend stop-backend status dataset-stats demo help

# By default, show help
.DEFAULT_GOAL := help