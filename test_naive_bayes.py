# naive_bayes_enhanced.py  (UI dengan Pilihan Fitur)
import streamlit as st
import pandas as pd
import numpy as np
import re, string, unicodedata
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             confusion_matrix)
import nltk, pickle, os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- one-time NLTK setup -------------------------------------------------------
@st.cache_resource
def nltk_setup():
    for pkg in ('punkt', 'stopwords', 'punkt_tab'):
        nltk.download(pkg, quiet=True)
nltk_setup()

# --- Indonesian stop-words -----------------------------------------------------
STOPWORDS = set(stopwords.words('indonesian')) if 'indonesian' in stopwords.fileids() \
            else set(stopwords.words('english'))
STEMMER = PorterStemmer()

# --- helper functions ---------------------------------------------------------
def clean_noise(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", '', text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove emojis / non-ascii
    text = re.sub(r"\s+", " ", text)
    return text.strip()

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

# --- auto-labeling -----------------------------------------------------------
def auto_label(text):
    txt = str(text).lower()
    
    # Keywords negatif
    negative_keywords = [
        "#indonesiagelap", "#indonesiagelap2025", "#kaburajadulu", "#peringatandarurat",
        "buruk", "jelek", "kecewa", "marah", "benci", "sedih", "susah", "sulit", 
        "gagal", "rusak", "hancur", "parah", "mengerikan", "tidak suka", "bosan"
    ]
    
    # Keywords positif
    positive_keywords = [
        "#positif", "#bahagia", "#sukses", "#syukur", "suka", "senang", "bagus", 
        "hebat", "mantap", "keren", "amazing", "wonderful", "excellent", "good",
        "baik", "indah", "cantik", "ganteng", "love", "cinta", "sayang", "happy",
        "terima kasih", "thanks", "grateful", "puas", "berhasil", "menang"
    ]
    
    # Hitung score
    negative_score = sum(1 for word in negative_keywords if word in txt)
    positive_score = sum(1 for word in positive_keywords if word in txt)
    
    if positive_score > negative_score:
        return "positive"
    elif negative_score > positive_score:
        return "negative"
    else:
        return "neutral"

# --- Streamlit UI -------------------------------------------------------------
st.set_page_config(page_title="Naïve Bayes Sentiment – Indonesia Twitter",
                   layout="wide")

# Header dengan tabs untuk navigasi fitur
st.title("📊 Naïve Bayes Sentiment Analysis – Indonesia Twitter Data")
st.markdown("---")

# Tabs untuk berbagai fitur
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Data & Config", 
    "🔍 Preprocessing", 
    "🤖 Model Training", 
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
        
        # Load data
        df = pd.read_csv(uploaded, engine='python', on_bad_lines='skip')
        st.success(f"✅ Successfully loaded {len(df)} rows")
        
        # Column selection
        st.subheader("🎯 Column Selection")
        text_col = st.selectbox("Select text column:", df.columns)
        df = df[[text_col]].dropna().rename(columns={text_col: "text"})
        
        # Auto labeling
        with st.spinner("Auto-labeling with hashtag heuristic..."):
            df["label"] = df["text"].apply(auto_label)
    
    with col2:
        st.subheader("⚙️ Model Configuration")
        
        # Test split configuration
        test_ratio = st.slider("Test size ratio:", 0.10, 0.50, 0.20, 0.05)
        
        # TF-IDF configuration
        max_features = st.number_input("Max TF-IDF features:", 
                                     min_value=1000, max_value=10000, 
                                     value=5000, step=500)
        
        # Naive Bayes alpha range
        alpha_min = st.number_input("Alpha min:", 0.1, 2.0, 0.1, 0.1)
        alpha_max = st.number_input("Alpha max:", 0.1, 2.0, 2.0, 0.1)
        
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
            enable_stemming = st.checkbox("Enable stemming", value=True,
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
                st.success("✅ Preprocessing completed!")
                
                # Show word count statistics
                df["word_count"] = df["clean"].str.split().map(len)
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg words per text", f"{df['word_count'].mean():.1f}")
                col2.metric("Min words", f"{df['word_count'].min()}")
                col3.metric("Max words", f"{df['word_count'].max()}")

# --- TAB 3: MODEL TRAINING ---------------------------------------------------
with tab3:
    if 'df' not in locals():
        st.warning("⚠️ Please upload and configure data in the 'Data & Config' tab first.")
    elif not st.session_state.get('preprocessing_done', False):
        st.warning("⚠️ Please complete preprocessing in the 'Preprocessing' tab first.")
    else:
        st.header("🤖 Model Training & Evaluation")
        
        # Get processed dataframe from session state
        if 'df_processed' in st.session_state:
            df = st.session_state.df_processed
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🏋️ Training Configuration")
            st.write(f"**Dataset:** {len(df)} samples")
            st.write(f"**Test ratio:** {test_ratio*100:.0f}%")
            st.write(f"**Max features:** {max_features:,}")
            
            if st.button("🚀 Start Training", type="primary"):
                # Check if preprocessing is done
                if "clean" not in df.columns:
                    st.error("❌ Preprocessing not completed. Please go to 'Preprocessing' tab and apply preprocessing first.")
                    st.stop()
                
                with st.spinner("Training model..."):
                    # TF-IDF Vectorization
                    vectorizer = TfidfVectorizer(max_features=max_features)
                    X = vectorizer.fit_transform(df["clean"])
                    y = df["label"]
                    
                    # Train/test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_ratio, random_state=42, stratify=y
                    )
                    
                    # Training curve
                    alphas = np.arange(alpha_min, alpha_max + 0.1, 0.1)
                    train_scores, val_scores = [], []
                    
                    progress_bar = st.progress(0)
                    for i, a in enumerate(alphas):
                        m = MultinomialNB(alpha=a)
                        m.fit(X_train, y_train)
                        train_scores.append(accuracy_score(y_train, m.predict(X_train)))
                        val_scores.append(accuracy_score(y_test, m.predict(X_test)))
                        progress_bar.progress((i + 1) / len(alphas))
                    
                    best_idx = np.argmax(val_scores)
                    best_alpha = alphas[best_idx]
                    
                    # Train final model
                    clf = MultinomialNB(alpha=best_alpha).fit(X_train, y_train)
                    
                    # Store in session state
                    st.session_state.model_trained = True
                    st.session_state.clf = clf
                    st.session_state.vectorizer = vectorizer
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.train_scores = train_scores
                    st.session_state.val_scores = val_scores
                    st.session_state.alphas = alphas
                    st.session_state.best_alpha = best_alpha
                    
                    st.success(f"✅ Training completed! Best alpha: {best_alpha:.2f}")
        
        with col2:
            if st.session_state.get('model_trained', False):
                st.subheader("📈 Training Results")
                
                # Get results from session state
                clf = st.session_state.clf
                y_test = st.session_state.y_test
                X_test = st.session_state.X_test
                
                # Predictions
                y_pred = clf.predict(X_test)
                
                # Metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Display metrics
                col_a, col_b = st.columns(2)
                col_a.metric("Accuracy", f"{acc:.3f}")
                col_a.metric("Precision", f"{prec:.3f}")
                col_b.metric("Recall", f"{rec:.3f}")
                col_b.metric("F1-Score", f"{f1:.3f}")
                
                # Training curve
                fig_curve, ax_curve = plt.subplots(figsize=(8, 4))
                ax_curve.plot(st.session_state.alphas, st.session_state.train_scores, 
                             marker="o", label="Train", linewidth=2)
                ax_curve.plot(st.session_state.alphas, st.session_state.val_scores, 
                             marker="o", label="Validation", linewidth=2)
                ax_curve.axvline(st.session_state.best_alpha, color="red", 
                               linestyle="--", label=f"Best α={st.session_state.best_alpha:.2f}")
                ax_curve.set_xlabel("Alpha")
                ax_curve.set_ylabel("Accuracy")
                ax_curve.legend()
                ax_curve.set_title("Training Curve")
                ax_curve.grid(True, alpha=0.3)
                st.pyplot(fig_curve)

# --- TAB 4: ANALYSIS & VISUALIZATION -----------------------------------------
with tab4:
    if not st.session_state.get('model_trained', False):
        st.warning("⚠️ Please train the model in the 'Model Training' tab first.")
    else:
        st.header("📈 Analysis & Visualization")
        
        # Get processed dataframe from session state
        if 'df_processed' in st.session_state:
            df = st.session_state.df_processed
        
        # Get model results
        clf = st.session_state.clf
        y_test = st.session_state.y_test
        X_test = st.session_state.X_test
        y_pred = clf.predict(X_test)
        
        # Analysis options
        analysis_type = st.selectbox("Choose analysis type:", [
            "Confusion Matrix",
            "Label Distribution", 
            "Word Length Distribution",
            "Word Clouds",
            "Classification Report"
        ])
        
        if analysis_type == "Confusion Matrix":
            st.subheader("🔍 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=clf.classes_, yticklabels=clf.classes_, ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        
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
            selected_sentiment = st.selectbox("Select sentiment:", clf.classes_)
            
            texts = " ".join(df[df["label"] == selected_sentiment]["clean"])
            if texts.strip():
                wc = WordCloud(background_color="white", colormap="tab10",
                              width=800, height=400).generate(texts)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                ax.set_title(f"Word Cloud - {selected_sentiment.title()} Sentiment")
                st.pyplot(fig)
            else:
                st.warning(f"No data available for {selected_sentiment} sentiment.")
        
        elif analysis_type == "Classification Report":
            st.subheader("📋 Detailed Classification Report")
            report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3))

# --- TAB 5: LIVE PREDICTION --------------------------------------------------
with tab5:
    if not st.session_state.get('model_trained', False):
        st.warning("⚠️ Please train the model in the 'Model Training' tab first.")
    else:
        st.header("🔮 Live Sentiment Prediction")
        
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
            show_processing = st.checkbox("Show text processing steps & debug info", value=False)
            
            st.subheader("🔧 Quick Fixes")
            if st.button("🔄 Retrain with Better Labels"):
                st.info("💡 Tip: If predictions seem wrong, consider:")
                st.write("- Upload more balanced training data")
                st.write("- Check if your text contains words similar to training data")
                st.write("- Adjust preprocessing settings")
                st.write("- Add more positive/negative keywords to auto-labeling")
        
        if predict_btn and new_text.strip():
            # Check if model is trained and preprocessing settings are available
            if 'vectorizer' not in st.session_state or 'clf' not in st.session_state:
                st.error("❌ Model not found. Please train the model first.")
                st.stop()
            
            # Get preprocessing settings from session or use defaults
            enable_cleaning = True
            enable_stemming = True  
            enable_stopwords = True
            
            with st.spinner("Analyzing sentiment..."):
                # Process text
                clean_text = full_pipeline(new_text, enable_cleaning, enable_stemming, enable_stopwords)
                
                # Vectorize and predict
                vectorizer = st.session_state.vectorizer
                clf = st.session_state.clf
                
                vec = vectorizer.transform([clean_text])
                pred = clf.predict(vec)[0]
                proba = clf.predict_proba(vec)[0]
                
                # Map to sentiment
                sentiment_map = {"negative": "Negative 😞", "neutral": "Neutral 😐", "positive": "Positive 😊"}
                sentiment = sentiment_map.get(pred, "Neutral 😐")
                
                # Color coding
                color_map = {"negative": "#ff4b4b", "neutral": "#ffa500", "positive": "#00ff00"}
                color = color_map.get(pred, "#ffa500")
                
                # Display result
                st.markdown("### 🎯 Prediction Result")
                st.markdown(f"<h2 style='color: {color};'>{sentiment}</h2>", unsafe_allow_html=True)
                
                if show_confidence:
                    st.markdown("### 📊 Confidence Scores")
                    for i, class_name in enumerate(clf.classes_):
                        conf_pct = proba[i] * 100
                        st.write(f"**{sentiment_map.get(class_name, class_name).split()[0]}**: {conf_pct:.1f}%")
                        st.progress(proba[i])
                
                if show_processing:
                    st.markdown("### 🔍 Text Processing & Debug Info")
                    st.write("**Original text:**")
                    st.text(new_text)
                    st.write("**Processed text:**")
                    st.text(clean_text)
                    
                    # Debug info
                    st.write("**Debug Information:**")
                    st.write(f"- Text length: {len(new_text)} characters")
                    st.write(f"- Processed length: {len(clean_text)} characters")
                    st.write(f"- Words after processing: {len(clean_text.split())} words")
                    
                    # Show feature vector info
                    feature_names = vectorizer.get_feature_names_out()
                    vec_array = vec.toarray()[0]
                    non_zero_features = [(feature_names[i], vec_array[i]) for i in range(len(vec_array)) if vec_array[i] > 0]
                    
                    if non_zero_features:
                        st.write("**Features found in text:**")
                        for feature, score in sorted(non_zero_features, key=lambda x: x[1], reverse=True)[:10]:
                            st.write(f"- {feature}: {score:.4f}")
                    else:
                        st.warning("⚠️ No features found! This might explain unexpected results.")
                    
                    # Show all probabilities
                    st.write("**All class probabilities:**")
                    for i, class_name in enumerate(clf.classes_):
                        st.write(f"- {class_name}: {proba[i]:.4f}")
                    
                    # Training data insights
                    if 'df_processed' in st.session_state:
                        df_temp = st.session_state.df_processed
                        st.write("**Training data distribution:**")
                        for label in clf.classes_:
                            count = len(df_temp[df_temp['label'] == label])
                            percentage = count / len(df_temp) * 100
                            st.write(f"- {label}: {count} samples ({percentage:.1f}%)")
                        
                        # Check if similar words exist in training data
                        words_in_input = set(clean_text.split())
                        if words_in_input:
                            for label in clf.classes_:
                                label_texts = ' '.join(df_temp[df_temp['label'] == label]['clean'].tolist())
                                label_words = set(label_texts.split())
                                common_words = words_in_input.intersection(label_words)
                                if common_words:
                                    st.write(f"- Words found in {label} training data: {list(common_words)[:5]}")
                else:
                    st.info("💡 Enable 'Show text processing steps' to see detailed analysis.")
        
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
        
        cols = st.columns(len(samples))
        for i, sample in enumerate(samples):
            if cols[i].button(f"📝 Sample {i+1}", key=f"sample_{i}"):
                st.session_state.sample_text = sample
                st.rerun()
        
        if 'sample_text' in st.session_state:
            st.text_area("Sample text loaded:", st.session_state.sample_text, key="loaded_sample")
