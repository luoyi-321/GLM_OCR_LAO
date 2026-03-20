#!/bin/bash
# GLM-OCR Quick Pipeline Runner
# Usage: ./run_pipeline.sh [command]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "GLM-OCR Training Pipeline"
    echo ""
    echo "Usage: ./run_pipeline.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  setup       Install dependencies and download model"
    echo "  prepare     Prepare training data (creates sample structure)"
    echo "  train       Run LoRA training (default)"
    echo "  train-full  Run full fine-tuning"
    echo "  export      Export LoRA to merged model"
    echo "  infer       Run inference (requires: ./run_pipeline.sh infer image.png)"
    echo "  serve       Start vLLM inference server"
    echo "  all         Run complete pipeline"
    echo "  tensorboard Start TensorBoard to monitor training"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_pipeline.sh setup"
    echo "  ./run_pipeline.sh prepare"
    echo "  ./run_pipeline.sh train"
    echo "  ./run_pipeline.sh infer ./test_image.png"
    echo "  ./run_pipeline.sh all"
}

case "${1:-help}" in
    setup)
        print_header "Setting up GLM-OCR Environment"
        $PYTHON "$SCRIPT_DIR/glm_ocr_pipeline.py" setup
        ;;

    prepare)
        print_header "Preparing Training Data"
        if [ -n "$2" ]; then
            $PYTHON "$SCRIPT_DIR/glm_ocr_pipeline.py" prepare --source "$2"
        else
            $PYTHON "$SCRIPT_DIR/glm_ocr_pipeline.py" prepare
        fi
        ;;

    train)
        print_header "Starting LoRA Training"
        $PYTHON "$SCRIPT_DIR/glm_ocr_pipeline.py" train --mode lora
        ;;

    train-full)
        print_header "Starting Full Fine-Tuning"
        $PYTHON "$SCRIPT_DIR/glm_ocr_pipeline.py" train --mode full
        ;;

    export)
        print_header "Exporting Model"
        $PYTHON "$SCRIPT_DIR/glm_ocr_pipeline.py" export
        ;;

    infer)
        if [ -z "$2" ]; then
            print_error "Please specify an image: ./run_pipeline.sh infer image.png"
            exit 1
        fi
        print_header "Running Inference"
        $PYTHON "$SCRIPT_DIR/glm_ocr_pipeline.py" infer --image "$2"
        ;;

    serve)
        PORT="${2:-8000}"
        print_header "Starting Inference Server on port $PORT"
        $PYTHON "$SCRIPT_DIR/glm_ocr_pipeline.py" serve --port "$PORT"
        ;;

    all)
        print_header "Running Full Pipeline"
        if [ -n "$2" ]; then
            $PYTHON "$SCRIPT_DIR/glm_ocr_pipeline.py" all --source "$2"
        else
            $PYTHON "$SCRIPT_DIR/glm_ocr_pipeline.py" all
        fi
        ;;

    tensorboard)
        print_header "Starting TensorBoard"
        OUTPUT_DIR="$SCRIPT_DIR/outputs/glm_ocr_lora/runs"
        if [ ! -d "$OUTPUT_DIR" ]; then
            print_warning "No training logs found. Run training first."
            OUTPUT_DIR="$SCRIPT_DIR/outputs"
        fi
        tensorboard --logdir "$OUTPUT_DIR"
        ;;

    help|--help|-h)
        show_help
        ;;

    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
