#!/bin/bash

echo "=========================================="
echo "Multilingual IR - Semester VIII Pipeline"
echo "=========================================="

echo ""
echo "Step 1: Generate embeddings"
echo "----------------------------"
python generate_embeddings.py --test

echo ""
echo "Step 2: Mine hard negatives"
echo "----------------------------"
python negative_mining.py --test

echo ""
echo "Step 3: Fine-tune model"
echo "----------------------------"
python finetune.py --epochs 1 --test

echo ""
echo "Step 4: Run full evaluation"
echo "----------------------------"
python evaluate.py --test

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo ""
echo "To launch the Streamlit demo, run:"
echo "  streamlit run streamlit_app.py"
