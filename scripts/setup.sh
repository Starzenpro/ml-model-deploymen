#!/bin/bash

echo "🚀 Setting up ML Model Development Environment"
echo "================================================"

# Update pip
echo "📦 Updating pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p app/models
mkdir -p logs

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/*.sh
chmod +x scripts/*.py

# Train model if not exists
if [ ! -f "app/models/model.pkl" ]; then
    echo "🤖 Training initial model..."
    python scripts/train.py
else
    echo "✅ Model already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run 'make train' to retrain the model"
echo "  2. Run 'make run' to start the API"
echo "  3. Open http://localhost:8000/docs"
