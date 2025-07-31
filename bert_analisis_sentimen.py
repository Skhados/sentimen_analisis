import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import re
import warnings
import os
from torch.cuda.amp import autocast, GradScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import seaborn as sns
import pickle
from datetime import datetime
import json
import time

# Windows compatibility
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- NLTK setup ---
@st.cache_resource
def nltk_setup():
    for pkg in ('punkt', 'stopwords', 'punkt_tab'):
        nltk.download(pkg, quiet=True)
nltk_setup()

# --- Indonesian stop-words ---
STOPWORDS = set(stopwords.words('indonesian')) if 'indonesian' in stopwords.fileids() \
            else set(stopwords.words('english'))
STEMMER = PorterStemmer()

# --- Enhanced Auto-labeling function with debugging ---
def auto_label(text, debug=False):
    # Handle non-string inputs
    if not isinstance(text, str):
        if isinstance(text, bytes):
            try:
                text = text.decode('utf-8')
            except UnicodeDecodeError:
                text = text.decode('utf-8', errors='ignore')
        else:
            text = str(text)
    
    # Handle empty or null values
    if pd.isna(text) or text is None or text.strip() == '':
        if debug:
            st.write(f"⚠️ Empty text detected: '{text}'")
        return "neutral"
    
    original_text = text
    txt = text.lower()
    
    # Enhanced Keywords - lebih lengkap
    negative_keywords = [
        # Hashtags negatif
        "#indonesiagelap", "#indonesiagelap2025", "#kaburajadulu", "#peringatandarurat",
        # Kata negatif umum
        "buruk", "jelek", "kecewa", "marah", "benci", "sedih", "susah", "sulit", 
        "gagal", "rusak", "hancur", "parah", "mengerikan", "tidak suka", "bosan",
        # Kata negatif tambahan
        "terrible", "awful", "bad", "worst", "hate", "angry", "disappointed", "sad",
        "difficult", "hard", "broken", "failed", "disaster", "horrible", "disgusting",
        "annoying", "frustrated", "upset", "worried", "scared", "afraid", "terrible",
        # Kata negatif Indonesia
        "jeburuk", "payah", "ampas", "zonk", "kecewa", "sebel", "kesel", "jengkel"
    ]
    
    positive_keywords = [
        # Hashtags positif
        "#positif", "#bahagia", "#sukses", "#syukur", "#blessed", "#happy", "#love",
        # Kata positif umum
        "suka", "senang", "bagus", "hebat", "mantap", "keren", "amazing", "wonderful", 
        "excellent", "good", "baik", "indah", "cantik", "ganteng", "love", "cinta", 
        "sayang", "happy", "terima kasih", "thanks", "grateful", "puas", "berhasil", "menang",
        # Kata positif tambahan
        "great", "awesome", "fantastic", "brilliant", "perfect", "beautiful", "nice",
        "super", "outstanding", "magnificent", "marvelous", "incredible", "impressive",
        "satisfied", "pleased", "delighted", "thrilled", "excited", "joyful", "cheerful",
        # Kata positif Indonesia
        "oke", "okeh", "mantul", "kece", "asik", "asyik", "top", "joss", "juara",
        "recommended", "recommend", "worthit", "worth it", "terbaik", "terbagus"
    ]
    
    # Hitung score dengan case-insensitive matching
    negative_matches = []
    positive_matches = []
    
    for word in negative_keywords:
        if word in txt:
            negative_matches.append(word)
    
    for word in positive_keywords:
        if word in txt:
            positive_matches.append(word)
    
    negative_score = len(negative_matches)
    positive_score = len(positive_matches)
    
    # Debug information
    if debug:
        st.write(f"**Text:** {original_text[:100]}...")
        st.write(f"**Lowercase:** {txt[:100]}...")
        st.write(f"**Positive matches:** {positive_matches} (Score: {positive_score})")
        st.write(f"**Negative matches:** {negative_matches} (Score: {negative_score})")
    
    # Determine sentiment
    if positive_score > negative_score:
        result = "positive"
    elif negative_score > positive_score:
        result = "negative"
    else:
        result = "neutral"
    
    if debug:
        st.write(f"**Final label:** {result}")
        st.write("---")
    
    return result

# --- Preprocessing functions ---
def clean_noise(text) -> str:
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8')
        except UnicodeDecodeError:
            text = text.decode('utf-8', errors='ignore')
    
    if not isinstance(text, str):
        text = str(text)
    
    if pd.isna(text) or text is None:
        return ""
    
    try:
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r"@\w+", '', text)
        text = re.sub(r"#\w+", '', text)  # Remove hashtags for processing
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    except Exception as e:
        st.warning(f"Text cleaning failed for: {text[:50]}... Error: {str(e)}")
        return str(text).strip()

def preprocess_text(text, remove_stopwords=True, apply_stemming=True):
    """Advanced text preprocessing"""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    
    # Apply stemming
    if apply_stemming:
        tokens = [STEMMER.stem(token) for token in tokens]
    
    return ' '.join(tokens)

def case_folding(text: str) -> str:
    return text.lower()

def tokenize(text: str):
    return word_tokenize(text)

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS and t not in string.punctuation]

def stemming(tokens):
    return [STEMMER.stem(t) for t in tokens]

def full_pipeline(text: str, enable_cleaning=True, enable_stemming=True, enable_stopwords=True) -> str:
    if enable_cleaning:
        text = clean_noise(text)
    text = case_folding(text)
    tokens = tokenize(text)
    if enable_stopwords:
        tokens = remove_stopwords(tokens)
    if enable_stemming:
        tokens = stemming(tokens)
    return " ".join(tokens)

# Safe CSV loading function
def load_csv_safely(uploaded_file):
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding, engine='python', on_bad_lines='skip')
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
            
            st.success(f"✅ Successfully loaded with {encoding} encoding")
            return df
            
        except Exception as e:
            st.warning(f"Failed with {encoding} encoding: {str(e)}")
            continue
    
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore', engine='python', on_bad_lines='skip')
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        st.warning("⚠️ Loaded with error handling - some characters may be corrupted")
        return df
        
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {str(e)}")
        return None

# --- BERT Model Classes ---
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Handle both pandas Series and numpy arrays
        if hasattr(self.texts, 'iloc'):
            text = str(self.texts.iloc[idx])
        else:
            text = str(self.texts[idx])
        
        if hasattr(self.labels, 'iloc'):
            label = self.labels.iloc[idx]
        else:
            label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTSentimentModel(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

# --- Training Functions ---
def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    """Train BERT model with mixed precision and validation"""
    scaler = GradScaler()
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.empty()
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress
            batch_progress = (batch_idx + 1) / len(train_loader)
            epoch_progress = (epoch + batch_progress) / num_epochs
            progress_bar.progress(epoch_progress)
            
            status_text.text(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Display metrics
        with metrics_container.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Train Loss", f"{avg_train_loss:.4f}")
            with col2:
                st.metric("Val Loss", f"{avg_val_loss:.4f}")
            with col3:
                st.metric("Val Accuracy", f"{val_accuracy:.4f}")
    
    return train_losses, val_losses, val_accuracies

def predict_sentiment(model, tokenizer, texts, label_encoder, device, batch_size=32):
    """Predict sentiment for new texts"""
    model.eval()
    predictions = []
    probabilities = []
    
    progress_bar = st.progress(0)
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
            progress_bar.progress((i + batch_size) / len(texts))
    
    # Convert predictions back to labels
    predicted_labels = label_encoder.inverse_transform(predictions)
    
    return predicted_labels, probabilities

# --- Visualization Functions ---
def create_wordcloud(df, sentiment_filter=None):
    """Create word cloud from text data"""
    if sentiment_filter:
        texts = df[df['label'] == sentiment_filter]['clean']
    else:
        texts = df['clean']
    
    text = ' '.join(texts.astype(str))
    
    if not text.strip():
        return None
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

def plot_training_metrics(train_losses, val_losses, val_accuracies):
    """Plot training metrics"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Training & Validation Loss', 'Validation Accuracy'),
        vertical_spacing=0.1
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(y=train_losses, name='Training Loss', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=val_losses, name='Validation Loss', line=dict(color='red')),
        row=1, col=1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(y=val_accuracies, name='Validation Accuracy', line=dict(color='green')),
        row=2, col=1
    )
    
    fig.update_layout(height=600, title_text="Training Metrics")
    return fig

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        aspect="auto",
        title="Confusion Matrix"
    )
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
            )
    
    return fig

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="🤖 BERT Sentiment Analysis – Indonesia Twitter",
        layout="wide"
    )
    
    # Header dengan tabs untuk navigasi fitur
    st.title("🤖 BERT Sentiment Analysis – Indonesia Twitter Data")
    st.markdown("---")
    
    # Tabs untuk berbagai fitur
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Data & Config", 
        "🔍 Preprocessing", 
        "🤖 BERT Training", 
        "📈 Analysis & Visualization", 
        "🔮 Live Prediction"
    ])
    
    # --- TAB 1: DATA & CONFIGURATION ----------------------------------------------
    with tab1:
        st.header("📁 Data Upload & Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📂 Dataset")
            uploaded = st.file_uploader("Upload CSV file", type="csv")
            
            if uploaded is None:
                st.warning("⚠️ Please upload a CSV file to begin analysis.")
                st.info("💡 Your CSV file should contain text data for sentiment analysis.")
                st.stop()
            
            # Load data safely
            df = load_csv_safely(uploaded)
            if df is None:
                st.stop()
            
            st.success(f"✅ Successfully loaded {len(df)} rows")
            
            # Column selection
            st.subheader("🎯 Column Selection")
            text_col = st.selectbox("Select text column:", df.columns)
            df = df[[text_col]].dropna().rename(columns={text_col: "text"})
            
            # Auto labeling
            with st.spinner("Auto-labeling with hashtag heuristic..."):
                df["label"] = df["text"].apply(auto_label)
        
        with col2:
            st.subheader("⚙️ BERT Configuration")
            
            # Model selection
            model_name = st.selectbox(
                "Choose BERT model:",
                [
                    "indobenchmark/indobert-base-p1",
                    "bert-base-uncased",
                    "bert-base-multilingual-cased",
                    "distilbert-base-uncased"
                ]
            )
            
            # Training configuration
            max_length = st.slider("Max sequence length:", 128, 512, 256)
            batch_size = st.selectbox("Batch size:", [8, 16, 32], index=1)
            learning_rate = st.selectbox("Learning rate:", [1e-5, 2e-5, 3e-5, 5e-5], index=1)
            num_epochs = st.slider("Number of epochs:", 1, 10, 3)
            dropout = st.slider("Dropout rate:", 0.1, 0.5, 0.3)
            
            # Test split configuration
            test_ratio = st.slider("Test size ratio:", 0.10, 0.50, 0.20, 0.05)
            
            st.subheader("📊 Data Overview")
            if 'df' in locals():
                label_counts = df["label"].value_counts()
                st.write("**Label Distribution:**")
                for label, count in label_counts.items():
                    st.write(f"• {label.title()}: {count} ({count/len(df)*100:.1f}%)")
        
        # Data preview
        if 'df' in locals():
            with st.expander("👀 Data Preview (First 10 rows)"):
                st.dataframe(df.head(10))
    
    # --- TAB 2: PREPROCESSING -----------------------------------------------------
    with tab2:
        if 'df' not in locals():
            st.warning("⚠️ Please upload and configure data in the 'Data & Config' tab first.")
        else:
            st.header("🔍 Text Preprocessing Configuration")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("🛠️ Preprocessing Options")
                enable_cleaning = st.checkbox("Enable noise cleaning", value=True, 
                                             help="Remove URLs, mentions, emojis")
                enable_stopwords = st.checkbox("Remove stopwords", value=True,
                                              help="Remove common words like 'dan', 'yang', etc.")
                enable_stemming = st.checkbox("Enable stemming", value=False,
                                             help="Reduce words to root form")
                
                st.subheader("📝 Preview Sample")
                if st.button("🎲 Get Random Sample"):
                    st.session_state.sample_idx = np.random.randint(0, len(df))
                
                if 'sample_idx' not in st.session_state:
                    st.session_state.sample_idx = 0
            
            with col2:
                st.subheader("🔍 Preprocessing Steps Demo")
                sample = df["text"].iloc[st.session_state.sample_idx]
                
                steps = []
                steps.append(("Original", sample))
                
                current_text = sample
                if enable_cleaning:
                    current_text = clean_noise(current_text)
                    steps.append(("After noise cleaning", current_text))
                
                current_text = case_folding(current_text)
                steps.append(("After case folding", current_text))
                
                tokens = tokenize(current_text)
                steps.append(("After tokenization", str(tokens)))
                
                if enable_stopwords:
                    tokens = remove_stopwords(tokens)
                    steps.append(("After stopword removal", str(tokens)))
                
                if enable_stemming:
                    tokens = stemming(tokens)
                    steps.append(("After stemming", str(tokens)))
                
                final_text = " ".join(tokens)
                steps.append(("Final result", final_text))
                
                for i, (step_name, step_result) in enumerate(steps):
                    if i == 0:
                        st.text_area(f"**{step_name}:**", step_result, height=60, key=f"step_{i}")
                    else:
                        with st.expander(f"{i}️⃣ {step_name}"):
                            st.code(step_result)
            
            # Apply preprocessing
            if st.button("🚀 Apply Preprocessing to All Data", type="primary"):
                with st.spinner("Processing all texts..."):
                    df["clean"] = df["text"].astype(str).apply(
                        lambda x: full_pipeline(x, enable_cleaning, enable_stemming, enable_stopwords)
                    )
                    st.session_state.preprocessing_done = True
                    st.session_state.df_processed = df  # Store processed dataframe
                    st.session_state.preprocessing_config = {
                        'enable_cleaning': enable_cleaning,
                        'enable_stemming': enable_stemming,
                        'enable_stopwords': enable_stopwords
                    }
                    st.success("✅ Preprocessing completed!")
                    
                    # Show word count statistics
                    df["word_count"] = df["clean"].str.split().map(len)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg words per text", f"{df['word_count'].mean():.1f}")
                    col2.metric("Min words", f"{df['word_count'].min()}")
                    col3.metric("Max words", f"{df['word_count'].max()}")
    
    # --- TAB 3: BERT TRAINING ----------------------------------------------------
    with tab3:
        if 'df' not in locals():
            st.warning("⚠️ Please upload and configure data in the 'Data & Config' tab first.")
        elif not st.session_state.get('preprocessing_done', False):
            st.warning("⚠️ Please complete preprocessing in the 'Preprocessing' tab first.")
        else:
            st.header("🤖 BERT Model Training & Evaluation")
            
            # Get processed dataframe from session state
            if 'df_processed' in st.session_state:
                df = st.session_state.df_processed
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🏋️ Training Configuration")
                st.write(f"**Dataset:** {len(df)} samples")
                st.write(f"**Model:** {model_name}")
                st.write(f"**Max length:** {max_length}")
                st.write(f"**Batch size:** {batch_size}")
                st.write(f"**Learning rate:** {learning_rate}")
                st.write(f"**Epochs:** {num_epochs}")
                st.write(f"**Test ratio:** {test_ratio*100:.0f}%")
                
                if st.button("🚀 Start BERT Training", type="primary"):
                    # Check if preprocessing is done
                    if "clean" not in df.columns:
                        st.error("❌ Preprocessing not completed. Please go to 'Preprocessing' tab and apply preprocessing first.")
                        st.stop()
                    
                    with st.spinner("Preparing data and starting BERT training..."):
                        
                        # Prepare data
                        texts = df["clean"]
                        labels = df["label"]
                        
                        # Check label distribution
                        label_counts = labels.value_counts()
                        if len(label_counts) < 2:
                            st.error("❌ Need at least 2 different sentiment classes for training!")
                            st.stop()
                        
                        # Encode labels
                        label_encoder = LabelEncoder()
                        encoded_labels = label_encoder.fit_transform(labels)
                        
                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(
                            texts, encoded_labels, test_size=test_ratio, random_state=42, stratify=encoded_labels
                        )
                        
                        # Train-validation split
                        val_size = 0.1
                        val_size_adjusted = val_size / (1 - test_ratio)
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train, y_train, test_size=val_size_adjusted, random_state=42, stratify=y_train
                        )
                        
                        # Convert numpy arrays back to pandas Series for compatibility
                        X_train = pd.Series(X_train.values if hasattr(X_train, 'values') else X_train, name='text')
                        X_val = pd.Series(X_val.values if hasattr(X_val, 'values') else X_val, name='text')
                        X_test = pd.Series(X_test.values if hasattr(X_test, 'values') else X_test, name='text')
                        y_train = pd.Series(y_train, name='label')
                        y_val = pd.Series(y_val, name='label')
                        y_test = pd.Series(y_test, name='label')
                        
                        st.info(f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
                        
                        # Initialize tokenizer and model
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = BERTSentimentModel(model_name, len(label_encoder.classes_), dropout).to(device)
                        
                        # Create datasets and dataloaders
                        train_dataset = SentimentDataset(X_train, y_train, tokenizer, max_length)
                        val_dataset = SentimentDataset(X_val, y_val, tokenizer, max_length)
                        
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size)
                        
                        # Initialize optimizer and scheduler
                        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
                        
                        total_steps = len(train_loader) * num_epochs
                        warmup_steps = 100
                        scheduler = get_linear_schedule_with_warmup(
                            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
                        )
                        
                        # Start training
                        st.info("🏋️‍♂️ Starting BERT training...")
                        
                        train_losses, val_losses, val_accuracies = train_model(
                            model, train_loader, val_loader, optimizer, scheduler, num_epochs, device
                        )
                        
                        # Store training results
                        st.session_state.update({
                            'trained_model': model,
                            'tokenizer': tokenizer,
                            'label_encoder': label_encoder,
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'val_accuracies': val_accuracies,
                            'X_test': X_test,
                            'y_test': y_test,
                            'bert_config': {
                                'model_name': model_name,
                                'max_length': max_length,
                                'batch_size': batch_size,
                                'learning_rate': learning_rate,
                                'num_epochs': num_epochs
                            }
                        })
                        
                        st.success("🎉 BERT training completed!")
            
            with col2:
                if st.session_state.get('trained_model'):
                    st.subheader("📈 Training Results")
                    
                    # Get results from session state
                    model = st.session_state.trained_model
                    tokenizer = st.session_state.tokenizer
                    label_encoder = st.session_state.label_encoder
                    
                    # Show training metrics
                    if 'train_losses' in st.session_state:
                        train_losses = st.session_state.train_losses
                        val_losses = st.session_state.val_losses
                        val_accuracies = st.session_state.val_accuracies
                        
                        # Display final metrics
                        col_a, col_b = st.columns(2)
                        col_a.metric("Final Train Loss", f"{train_losses[-1]:.4f}")
                        col_a.metric("Final Val Loss", f"{val_losses[-1]:.4f}")
                        col_b.metric("Final Val Accuracy", f"{val_accuracies[-1]:.4f}")
                        col_b.metric("Best Val Accuracy", f"{max(val_accuracies):.4f}")
                        
                        # Training curve
                        fig_curve = plot_training_metrics(train_losses, val_losses, val_accuracies)
                        st.plotly_chart(fig_curve, use_container_width=True)
    
    # --- TAB 4: ANALYSIS & VISUALIZATION -----------------------------------------
    with tab4:
        if not st.session_state.get('trained_model'):
            st.warning("⚠️ Please train the BERT model in the 'BERT Training' tab first.")
        else:
            st.header("📈 Analysis & Visualization")
            
            # Get processed dataframe from session state
            if 'df_processed' in st.session_state:
                df = st.session_state.df_processed
            
            # Get model results
            model = st.session_state.trained_model
            tokenizer = st.session_state.tokenizer
            label_encoder = st.session_state.label_encoder
            
            # Analysis options
            analysis_type = st.selectbox("Choose analysis type:", [
                "Training Metrics",
                "Test Set Evaluation", 
                "Label Distribution",
                "Word Length Distribution",
                "Word Clouds",
                "Confusion Matrix"
            ])
            
            if analysis_type == "Training Metrics":
                st.subheader("📈 Training Metrics Visualization")
                
                if 'train_losses' in st.session_state:
                    train_losses = st.session_state.train_losses
                    val_losses = st.session_state.val_losses
                    val_accuracies = st.session_state.val_accuracies
                    
                    fig_metrics = plot_training_metrics(train_losses, val_losses, val_accuracies)
                    st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    # Training summary
                    st.write("**Training Summary:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Final Train Loss", f"{train_losses[-1]:.4f}")
                    with col2:
                        st.metric("Final Val Loss", f"{val_losses[-1]:.4f}")
                    with col3:
                        st.metric("Best Val Accuracy", f"{max(val_accuracies):.4f}")
                else:
                    st.warning("⚠️ No training metrics available.")
            
            elif analysis_type == "Test Set Evaluation":
                st.subheader("🧪 Test Set Evaluation")
                
                if 'X_test' in st.session_state and 'y_test' in st.session_state:
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test
                    
                    if st.button("📊 Evaluate on Test Set"):
                        with st.spinner("Evaluating model on test set..."):
                            test_texts = X_test.tolist()
                            predictions, probabilities = predict_sentiment(
                                model, tokenizer, test_texts, label_encoder, device
                            )
                            
                            # Convert back to original labels
                            y_test_labels = label_encoder.inverse_transform(y_test)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test_labels, predictions)
                            report = classification_report(y_test_labels, predictions, output_dict=True)
                            
                            st.subheader("📈 Test Results")
                            
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.metric("Test Accuracy", f"{accuracy:.4f}")
                                
                                # Show classification report
                                st.write("**Classification Report:**")
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.round(4))
                            
                            with col2:
                                # Confusion matrix
                                fig_cm = plot_confusion_matrix(y_test_labels, predictions, label_encoder.classes_)
                                st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    st.warning("⚠️ No test set available. Please retrain the model.")
            
            elif analysis_type == "Label Distribution":
                st.subheader("📊 Label Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar plot
                    label_counts = df["label"].value_counts()
                    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=label_counts.index, y=label_counts.values, 
                               palette="Set2", ax=ax_bar)
                    ax_bar.set_title("Label Distribution")
                    ax_bar.set_ylabel("Count")
                    st.pyplot(fig_bar)
                
                with col2:
                    # Pie chart
                    sentiment_map = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
                    df["sentiment"] = df["label"].map(sentiment_map).fillna("Neutral")
                    sent_counts = df["sentiment"].value_counts()
                    
                    fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                    ax_pie.pie(sent_counts.values, labels=sent_counts.index,
                              autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3"))
                    ax_pie.set_title("Sentiment Distribution")
                    st.pyplot(fig_pie)
            
            elif analysis_type == "Word Length Distribution":
                st.subheader("📏 Word Length Distribution")
                if "word_count" not in df.columns:
                    df["word_count"] = df["clean"].str.split().map(len)
                
                fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
                sns.histplot(data=df, x="word_count", hue="label", bins=30, kde=True, ax=ax_hist)
                ax_hist.set_xlabel("Number of words")
                ax_hist.set_ylabel("Frequency")
                ax_hist.set_title("Word Count Distribution by Sentiment")
                st.pyplot(fig_hist)
            
            elif analysis_type == "Word Clouds":
                st.subheader("☁️ Word Clouds by Sentiment")
                selected_sentiment = st.selectbox("Select sentiment:", label_encoder.classes_)
                
                fig = create_wordcloud(df, selected_sentiment)
                if fig:
                    st.pyplot(fig)
                else:
                    st.warning(f"No data available for {selected_sentiment} sentiment.")
            
            elif analysis_type == "Confusion Matrix":
                st.subheader("🔍 Confusion Matrix")
                st.info("💡 Please run 'Test Set Evaluation' first to generate confusion matrix.")
    
    # --- TAB 5: LIVE PREDICTION --------------------------------------------------
    with tab5:
        if not st.session_state.get('trained_model'):
            st.warning("⚠️ Please train the BERT model in the 'BERT Training' tab first.")
        else:
            st.header("🔮 Live Sentiment Prediction")
            
            model = st.session_state.trained_model
            tokenizer = st.session_state.tokenizer
            label_encoder = st.session_state.label_encoder
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📝 Input Text")
                new_text = st.text_area("Enter your text for sentiment analysis:", 
                                       height=150, 
                                       placeholder="Type your tweet or text here...")
                
                col_a, col_b = st.columns(2)
                predict_btn = col_a.button("🎯 Predict Sentiment", type="primary")
                clear_btn = col_b.button("🧹 Clear Text")
                
                if clear_btn:
                    st.rerun()
            
            with col2:
                st.subheader("⚙️ Prediction Settings")
                show_confidence = st.checkbox("Show confidence scores", value=True)
                show_processing = st.checkbox("Show text processing steps", value=False)
                
                st.subheader("🔧 Model Info")
                if 'bert_config' in st.session_state:
                    config = st.session_state.bert_config
                    st.write(f"**Model:** {config['model_name']}")
                    st.write(f"**Max length:** {config['max_length']}")
                    st.write(f"**Classes:** {len(label_encoder.classes_)}")
            
            if predict_btn and new_text.strip():
                with st.spinner("Analyzing sentiment with BERT..."):
                    # Get preprocessing settings from session or use defaults
                    preprocessing_config = st.session_state.get('preprocessing_config', {
                        'enable_cleaning': True,
                        'enable_stemming': False,
                        'enable_stopwords': True
                    })
                    
                    # Process text
                    clean_text = full_pipeline(
                        new_text, 
                        preprocessing_config['enable_cleaning'], 
                        preprocessing_config['enable_stemming'], 
                        preprocessing_config['enable_stopwords']
                    )
                    
                    # Predict using BERT
                    predictions, probabilities = predict_sentiment(
                        model, tokenizer, [clean_text], label_encoder, device
                    )
                    
                    pred = predictions[0]
                    proba = probabilities[0]
                    
                    # Map to sentiment
                    sentiment_map = {"negative": "Negative 😞", "neutral": "Neutral 😐", "positive": "Positive 😊"}
                    sentiment = sentiment_map.get(pred, "Neutral 😐")
                    
                    # Color coding
                    color_map = {"negative": "#ff4b4b", "neutral": "#ffa500", "positive": "#00ff00"}
                    color = color_map.get(pred, "#ffa500")
                    
                    # Display result
                    st.markdown("### 🎯 BERT Prediction Result")
                    st.markdown(f"<h2 style='color: {color};'>{sentiment}</h2>", unsafe_allow_html=True)
                    
                    if show_confidence:
                        st.markdown("### 📊 Confidence Scores")
                        for i, class_name in enumerate(label_encoder.classes_):
                            conf_pct = proba[i] * 100
                            st.write(f"**{sentiment_map.get(class_name, class_name).split()[0]}**: {conf_pct:.1f}%")
                            st.progress(proba[i])
                    
                    if show_processing:
                        st.markdown("### 🔍 Text Processing Steps")
                        st.write("**Original text:**")
                        st.text(new_text)
                        st.write("**Processed text:**")
                        st.text(clean_text)
                        
                        # Debug info
                        st.write("**Processing Information:**")
                        st.write(f"- Text length: {len(new_text)} characters")
                        st.write(f"- Processed length: {len(clean_text)} characters")
                        st.write(f"- Words after processing: {len(clean_text.split())} words")
                        
                        # Show all probabilities
                        st.write("**All class probabilities:**")
                        for i, class_name in enumerate(label_encoder.classes_):
                            st.write(f"- {class_name}: {proba[i]:.4f}")
            
            elif predict_btn:
                st.warning("⚠️ Please enter some text to analyze.")
            
            # Sample texts for testing
            st.markdown("### 💡 Try these sample texts:")
            samples = [
                "Saya sangat bahagia hari ini! Mobil sport ini keren banget!",
                "Saya suka mobil sport yang bagus dan cepat",
                "Situasi ekonomi semakin memburuk #indonesiagelap",
                "Cuaca hari ini biasa saja, tidak ada yang istimewa",
                "Terima kasih atas dukungannya! Saya senang sekali #syukur",
                "Pemerintah harus lebih tegas #peringatandarurat",
                "Produk ini bagus sekali, saya sangat puas!",
                "Pelayanan buruk sekali, saya kecewa berat"
            ]
            
            cols = st.columns(min(len(samples), 4))  # Limit to 4 columns for better display
            for i, sample in enumerate(samples):
                col_idx = i % len(cols)
                if cols[col_idx].button(f"📝 Sample {i+1}", key=f"sample_{i}"):
                    st.session_state.sample_text = sample
                    st.rerun()
            
            if 'sample_text' in st.session_state:
                st.text_area("Sample text loaded:", st.session_state.sample_text, key="loaded_sample")

if __name__ == "__main__":
    main()