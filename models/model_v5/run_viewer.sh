#!/bin/bash
# Quick start script for Streamlit variant viewer

echo "🚀 Starting Streamlit variant viewer..."
echo ""
echo "If you haven't preprocessed yet, run:"
echo "  python preprocess.py"
echo ""
echo "The app will open at: http://localhost:8501"
echo ""

streamlit run streamlit_variant_viewer.py --logger.level=warning
