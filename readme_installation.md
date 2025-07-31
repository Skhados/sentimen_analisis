# 🚀 Sentiment Analysis Installation Guide

This guide helps you install all dependencies for both Naive Bayes and BERT sentiment analysis scripts.

## 📋 Requirements

- **Python 3.8+** (3.9+ recommended)
- **pip** (Python package installer)
- **8GB+ RAM** (16GB recommended for BERT)
- **Internet connection** (for downloading models)

## 🎯 Quick Installation

### Option 1: Automatic Setup (Recommended)
```bash
# Run the installation script
python install_setup.py

# Test the installation
streamlit run test_setup.py
```

### Option 2: Manual Installation
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install minimal version
pip install -r requirements_minimal.txt

# For GPU support (NVIDIA only)
pip install -r requirements_gpu.txt
```

### Option 3: Step-by-Step
```bash
# 1. Core dependencies
pip install streamlit pandas numpy scikit-learn

# 2. NLP libraries
pip install nltk transformers torch

# 3. Visualization
pip install matplotlib seaborn plotly wordcloud

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 📦 Package Categories

### 🎨 **UI & Visualization**
- `streamlit` - Web interface
- `matplotlib` - Basic plotting
- `seaborn` - Statistical plots
- `plotly` - Interactive charts
- `wordcloud` - Word cloud generation

### 🔢 **Data Processing**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Traditional ML algorithms

### 🧠 **Machine Learning**
- `torch` - PyTorch deep learning
- `transformers` - Hugging Face BERT models
- `tokenizers` - Fast text tokenization
- `accelerate` - Training optimization

### 📝 **Text Processing**
- `nltk` - Natural language toolkit
- `regex` - Advanced regular expressions
- `unidecode` - Unicode normalization

### ⚡ **Performance**
- `psutil` - System monitoring
- `joblib` - Parallel processing
- `cachetools` - Caching utilities
- `tqdm` - Progress bars

## 🖥️ System-Specific Instructions

### Windows
```cmd
# Install Microsoft Visual C++ if needed
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

pip install -r requirements.txt
```

### macOS
```bash
# Install Xcode command line tools if needed
xcode-select --install

pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt update
sudo apt install python3-pip python3-dev build-essential

pip install -r requirements.txt
```

## 🎮 GPU Support (Optional)

### NVIDIA GPU Setup
```bash
# Check CUDA compatibility
nvidia-smi

# Install CUDA version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### CPU-Only Setup
```bash
# Install CPU version (default)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 🧪 Testing Installation

### Quick Test
```bash
# Run the test script
streamlit run test_setup.py
```

### Manual Verification
```python
# Test in Python
import streamlit as st
import pandas as pd
import torch
import transformers
import nltk
import sklearn

print("✅ All packages imported successfully!")
print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
```

## 🚀 Running the Applications

### Naive Bayes Sentiment Analysis
```bash
streamlit run test_naive_bayes.py
```

### BERT Sentiment Analysis
```bash
streamlit run bert_analisis_sentimen.py
```

## 🐛 Troubleshooting

### Common Issues

#### 1. **PyTorch Installation Error**
```bash
# Try CPU version first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or specific CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 2. **NLTK Data Missing**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

#### 3. **Streamlit Port Issues**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

#### 4. **Memory Errors (BERT)**
- Reduce batch size in config
- Use CPU instead of GPU
- Close other applications
- Use DistilBERT instead of BERT

#### 5. **Model Download Failures**
```python
# Manual model download
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
```

## 📊 Version Compatibility

| Python | PyTorch | Transformers | Streamlit |
|--------|---------|-------------|-----------|
| 3.8    | 1.12+   | 4.21+       | 1.28+     |
| 3.9    | 1.13+   | 4.25+       | 1.30+     |
| 3.10   | 1.13+   | 4.30+       | 1.32+     |
| 3.11   | 2.0+    | 4.35+       | 1.35+     |

## 💡 Performance Tips

### For Better Speed:
- Use SSD storage
- Install CUDA version if you have NVIDIA GPU
- Increase batch size if you have more RAM
- Use `accelerate` library for training

### For Lower Memory Usage:
- Use DistilBERT instead of BERT
- Reduce max sequence length
- Lower batch size
- Close unnecessary applications

## 📞 Support

If you encounter issues:

1. **Check Python version**: `python --version`
2. **Update pip**: `pip install --upgrade pip`
3. **Clear pip cache**: `pip cache purge`
4. **Reinstall problematic package**: `pip uninstall package_name && pip install package_name`
5. **Check system resources**: Ensure sufficient RAM and disk space

## 🎉 Success Indicators

Installation is successful when:
- ✅ All imports work without errors
- ✅ Test script runs in Streamlit
- ✅ NLTK data downloads complete
- ✅ Model loading works (may take time on first run)
- ✅ GPU detection works (if applicable)

Happy analyzing! 🚀