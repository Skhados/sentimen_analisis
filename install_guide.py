#!/usr/bin/env python3
"""
Installation Guide and Setup Script for Sentiment Analysis Projects
================================================================

This script helps you install all required dependencies for both:
1. test_naive_bayes.py (Naive Bayes Sentiment Analysis)
2. bert_analisis_sentimen.py (BERT Sentiment Analysis)

Usage:
    python install_setup.py
    
Or manually install using pip:
    pip install -r requirements.txt
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(message, color=Colors.OKGREEN):
    """Print colored message to terminal"""
    print(f"{color}{message}{Colors.ENDC}")

def check_python_version():
    """Check if Python version is compatible"""
    print_colored("🐍 Checking Python version...", Colors.HEADER)
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_colored(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required!", Colors.FAIL)
        return False
    
    print_colored(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!", Colors.OKGREEN)
    return True

def install_package(package_name, description=""):
    """Install a single package with error handling"""
    try:
        print_colored(f"📦 Installing {package_name}... {description}", Colors.OKBLUE)
        subprocess.run([sys.executable, "-m", "pip", "install", package_name], 
                      check=True, capture_output=True, text=True)
        print_colored(f"✅ {package_name} installed successfully!", Colors.OKGREEN)
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Failed to install {package_name}: {e.stderr}", Colors.FAIL)
        return False

def install_torch():
    """Install PyTorch with appropriate CUDA support"""
    print_colored("🔥 Installing PyTorch...", Colors.HEADER)
    
    system = platform.system().lower()
    
    # Detect CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print_colored("✅ CUDA already available!", Colors.OKGREEN)
            return True
    except ImportError:
        pass
    
    # Install CPU version by default (more compatible)
    torch_packages = [
        "torch>=1.12.0",
        "torchvision>=0.13.0", 
        "torchaudio>=0.12.0"
    ]
    
    print_colored("💡 Installing CPU version of PyTorch (recommended for most users)", Colors.WARNING)
    print_colored("   For GPU support, install CUDA version manually after setup", Colors.WARNING)
    
    for package in torch_packages:
        if not install_package(package):
            return False
    
    return True

def install_requirements():
    """Install all required packages"""
    print_colored("📋 Installing Core Dependencies...", Colors.HEADER)
    
    # Core packages (essential for both scripts)
    core_packages = [
        ("streamlit>=1.28.0", "Web UI framework"),
        ("pandas>=1.5.0", "Data manipulation"),
        ("numpy>=1.21.0", "Numerical computing"),
        ("scikit-learn>=1.1.0", "Machine learning"),
        ("matplotlib>=3.5.0", "Basic plotting"),
        ("seaborn>=0.11.0", "Statistical visualization"),
        ("plotly>=5.10.0", "Interactive plots"),
        ("nltk>=3.8", "Natural language processing"),
        ("wordcloud>=1.9.0", "Word cloud generation"),
        ("regex>=2022.7.9", "Advanced regex"),
        ("tqdm>=4.64.0", "Progress bars"),
        ("requests>=2.28.0", "HTTP requests"),
        ("openpyxl>=3.0.10", "Excel file support"),
    ]
    
    print_colored("Installing core packages...", Colors.OKBLUE)
    for package, desc in core_packages:
        install_package(package, desc)
    
    # Install PyTorch
    install_torch()
    
    # Transformers for BERT
    print_colored("🤖 Installing Transformers for BERT...", Colors.HEADER)
    transformers_packages = [
        ("transformers>=4.21.0", "Hugging Face transformers"),
        ("tokenizers>=0.13.0", "Fast tokenizers"),
        ("accelerate>=0.21.0", "Training acceleration"),
    ]
    
    for package, desc in transformers_packages:
        install_package(package, desc)
    
    # Optional performance packages
    print_colored("⚡ Installing Performance Packages...", Colors.HEADER)
    optional_packages = [
        ("psutil>=5.9.0", "System monitoring"),
        ("joblib>=1.1.0", "Parallel processing"),
        ("cachetools>=5.2.0", "Caching utilities"),
        ("rich>=12.5.0", "Rich terminal output"),
    ]
    
    for package, desc in optional_packages:
        install_package(package, desc)

def download_nltk_data():
    """Download required NLTK data"""
    print_colored("📚 Downloading NLTK Data...", Colors.HEADER)
    
    try:
        import nltk
        
        # Required NLTK data for both scripts
        nltk_downloads = [
            'punkt',
            'stopwords', 
            'punkt_tab',
            'wordnet',
            'omw-1.4'
        ]
        
        for data in nltk_downloads:
            try:
                print_colored(f"📥 Downloading {data}...", Colors.OKBLUE)
                nltk.download(data, quiet=True)
                print_colored(f"✅ {data} downloaded!", Colors.OKGREEN)
            except Exception as e:
                print_colored(f"⚠️ Warning: Could not download {data}: {e}", Colors.WARNING)
                
    except ImportError:
        print_colored("❌ NLTK not installed. Please install it first.", Colors.FAIL)

def verify_installation():
    """Verify that all packages are properly installed"""
    print_colored("🔍 Verifying Installation...", Colors.HEADER)
    
    test_imports = [
        ("streamlit", "Streamlit UI"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("plotly", "Plotly"),
        ("nltk", "NLTK"),
        ("wordcloud", "WordCloud"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
    ]
    
    failed_imports = []
    
    for module, name in test_imports:
        try:
            __import__(module)
            print_colored(f"✅ {name} - OK", Colors.OKGREEN)
        except ImportError as e:
            print_colored(f"❌ {name} - FAILED: {e}", Colors.FAIL)
            failed_imports.append(name)
    
    if failed_imports:
        print_colored(f"\n⚠️ {len(failed_imports)} packages failed to import:", Colors.WARNING)
        for package in failed_imports:
            print_colored(f"  - {package}", Colors.WARNING)
        return False
    else:
        print_colored("\n🎉 All packages installed successfully!", Colors.OKGREEN)
        return True

def create_test_script():
    """Create a test script to verify everything works"""
    print_colored("📝 Creating test script...", Colors.HEADER)
    
    test_script = '''#!/usr/bin/env python3
"""
Test script to verify sentiment analysis setup
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import transformers
import nltk
import sklearn
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def main():
    st.title("🧪 Sentiment Analysis Setup Test")
    
    st.success("✅ All imports successful!")
    
    # Test basic functionality
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Package Versions")
        st.write(f"- Python: {sys.version}")
        st.write(f"- Streamlit: {st.__version__}")
        st.write(f"- Pandas: {pd.__version__}")
        st.write(f"- NumPy: {np.__version__}")
        st.write(f"- PyTorch: {torch.__version__}")
        st.write(f"- Transformers: {transformers.__version__}")
        st.write(f"- Scikit-learn: {sklearn.__version__}")
    
    with col2:
        st.subheader("🚀 System Info")
        st.write(f"- CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"- CUDA Device: {torch.cuda.get_device_name()}")
        st.write(f"- CPU Threads: {torch.get_num_threads()}")
    
    # Test sample data
    st.subheader("📝 Sample Data Test")
    sample_data = pd.DataFrame({
        'text': ['I love this!', 'This is terrible', 'Okay, not bad'],
        'label': ['positive', 'negative', 'neutral']
    })
    st.dataframe(sample_data)
    
    # Test plotting
    st.subheader("📈 Plotting Test")
    fig = px.bar(sample_data, x='label', title='Sample Sentiment Distribution')
    st.plotly_chart(fig)
    
    st.success("🎉 Setup verification complete! You're ready to run the sentiment analysis scripts.")

if __name__ == "__main__":
    import sys
    main()
'''
    
    with open('test_setup.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print_colored("✅ Test script created: test_setup.py", Colors.OKGREEN)
    print_colored("   Run: streamlit run test_setup.py", Colors.OKBLUE)

def main():
    """Main installation process"""
    print_colored("=" * 60, Colors.HEADER)
    print_colored("🚀 SENTIMENT ANALYSIS SETUP INSTALLER", Colors.HEADER)
    print_colored("=" * 60, Colors.HEADER)
    print_colored("This will install all dependencies for:", Colors.OKBLUE)
    print_colored("  • Naive Bayes Sentiment Analysis", Colors.OKBLUE)
    print_colored("  • BERT Sentiment Analysis", Colors.OKBLUE)
    print_colored("=" * 60, Colors.HEADER)
    
    # Check Python version
    if not check_python_version():
        print_colored("Please upgrade Python and try again.", Colors.FAIL)
        return
    
    # Ask for confirmation
    response = input(f"{Colors.WARNING}Continue with installation? (y/N): {Colors.ENDC}")
    if response.lower() not in ['y', 'yes']:
        print_colored("Installation cancelled.", Colors.WARNING)
        return
    
    try:
        # Install packages
        install_requirements()
        
        # Download NLTK data
        download_nltk_data()
        
        # Verify installation
        success = verify_installation()
        
        if success:
            # Create test script
            create_test_script()
            
            print_colored("\n" + "=" * 60, Colors.OKGREEN)
            print_colored("🎉 INSTALLATION COMPLETE!", Colors.OKGREEN)
            print_colored("=" * 60, Colors.OKGREEN)
            print_colored("Next steps:", Colors.OKBLUE)
            print_colored("1. Run test: streamlit run test_setup.py", Colors.OKBLUE)
            print_colored("2. Run Naive Bayes: streamlit run test_naive_bayes.py", Colors.OKBLUE)
            print_colored("3. Run BERT: streamlit run bert_analisis_sentimen.py", Colors.OKBLUE)
            print_colored("=" * 60, Colors.OKGREEN)
        else:
            print_colored("\n❌ Installation completed with errors.", Colors.FAIL)
            print_colored("Please check the error messages above and install missing packages manually.", Colors.WARNING)
            
    except KeyboardInterrupt:
        print_colored("\n\n⚠️ Installation interrupted by user.", Colors.WARNING)
    except Exception as e:
        print_colored(f"\n❌ Installation failed: {e}", Colors.FAIL)

if __name__ == "__main__":
    main()
