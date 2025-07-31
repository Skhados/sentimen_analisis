# combined_sentiment_app.py
import streamlit as st

# Sidebar: Pilihan Model
st.set_page_config(page_title="Sentiment Analysis: Naive Bayes vs BERT", layout="wide")
st.sidebar.title("⚙️ Pilih Metode Analisis")
model_choice = st.sidebar.radio("Metode Analisis Sentimen:", ["Naive Bayes", "BERT", "Perbandingan"], index=0)

# Menjalankan aplikasi sesuai pilihan
if model_choice == "Naive Bayes":
    st.title("📊 Naive Bayes Sentiment Analysis")
    st.info("Anda sedang menggunakan model Naive Bayes untuk analisis sentimen.")
    try:
        with open("test_naive_bayes.py", "rb") as f:
            exec(compile(f.read(), "test_naive_bayes.py", 'exec'))
        # Pastikan metrik dimasukkan ke session_state oleh file test_naive_bayes.py
        if not st.session_state.get("nb_metrics"):
            st.warning("⚠️ Metrik evaluasi Naive Bayes belum tersedia. Pastikan file test_naive_bayes.py menyimpan hasil ke st.session_state['nb_metrics'].")
    except Exception as e:
        st.error(f"Gagal memuat modul Naive Bayes: {str(e)}")

elif model_choice == "BERT":
    st.title("🤖 BERT Sentiment Analysis")
    st.info("Anda sedang menggunakan model BERT untuk analisis sentimen.")
    try:
        with open("bert_analisis_sentimen.py", "rb") as f:
            exec(compile(f.read(), "bert_analisis_sentimen.py", 'exec'))
        # Pastikan metrik dimasukkan ke session_state oleh file bert_analisis_sentimen.py
        if not st.session_state.get("bert_metrics"):
            st.warning("⚠️ Metrik evaluasi BERT belum tersedia. Pastikan file bert_analisis_sentimen.py menyimpan hasil ke st.session_state['bert_metrics'].")
    except Exception as e:
        st.error(f"Gagal memuat modul BERT: {str(e)}")

elif model_choice == "Perbandingan":
    st.title("📊 Perbandingan Naive Bayes vs BERT")
    st.info("Halaman ini menampilkan perbandingan hasil kedua model setelah dilatih.")

    nb_metrics = st.session_state.get("nb_metrics")
    bert_metrics = st.session_state.get("bert_metrics")

    if not nb_metrics or not bert_metrics:
        st.warning("⚠️ Silakan latih kedua model terlebih dahulu untuk melihat perbandingan.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 Naive Bayes")
            st.metric("Accuracy", f"{nb_metrics['accuracy']:.3f}")
            st.metric("Precision", f"{nb_metrics['precision']:.3f}")
            st.metric("Recall", f"{nb_metrics['recall']:.3f}")
            st.metric("F1 Score", f"{nb_metrics['f1']:.3f}")

        with col2:
            st.subheader("🤖 BERT")
            st.metric("Accuracy", f"{bert_metrics['accuracy']:.3f}")
            st.metric("Precision", f"{bert_metrics['precision']:.3f}")
            st.metric("Recall", f"{bert_metrics['recall']:.3f}")
            st.metric("F1 Score", f"{bert_metrics['f1']:.3f}")
