import streamlit as st
from transformers import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tempfile, os
import moviepy.editor as mp
import speech_recognition as sr
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# -------------------- CSS / UI --------------------
st.markdown("""
<style>
body { background-color: #fdf6f0; color: #333333; font-family: 'Helvetica', sans-serif; }
.stButton>button { border-radius: 10px; background-color: #FFB347; color: white; font-weight: bold; }
.stMetric { background-color: #fff0e6; border-radius: 15px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

pastel_colors = ['#FFB347','#FFCCBC','#B2DFDB','#C5CAE9','#F48FB1']

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    return sentiment_model, summarizer, emotion_model

sentiment_model, summarizer, emotion_model = load_models()

# ---------------- HELPERS ----------------
def detect_language(text):
    try: return detect(text) if len(text.strip())>2 else "en"
    except: return "en"

def translate_to_english(text, src_lang):
    return text if src_lang=="en" else GoogleTranslator(source=src_lang, target="en").translate(text)

def analyze_sentiment(text):
    lang = detect_language(text)
    translated = translate_to_english(text, lang)
    result = sentiment_model(translated)[0]
    label, score = result["label"], result["score"]
    emoji_map = {"Positive":"😀","Negative":"😡","Neutral":"😐"}
    return f"{label} {emoji_map.get(label,'')}", score, lang

def summarize_text(text):
    try: return summarizer(text, max_length=60, min_length=5, do_sample=False)[0]['summary_text']
    except: return text if len(text.split())<5 else "⚠️ Summary not available."

def analyze_emotion(text):
    scores = emotion_model(text)[0]
    sorted_scores = sorted(scores, key=lambda x:x['score'], reverse=True)
    top = sorted_scores[0]
    all_emotions = {s['label']:s['score'] for s in sorted_scores}
    return top['label'], top['score'], all_emotions

def generate_wordcloud(texts):
    combined_text = " ".join(texts)
    wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)
    plt.close()
    return wc

def generate_pdf(df, wc_img, title="e-Consultation Report"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width/2, height-50, title)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-100, f"📝 Total Feedback: {len(df)}")
    c.drawString(50, height-120, f"😊 Top Emotion: {df['top_emotion'].mode()[0]}")
    c.drawString(50, height-140, f"🌐 Most Active Language: {df['language'].mode()[0]}")
    c.drawString(50, height-170, "☁️ WordCloud of Feedback:")
    img_buffer = BytesIO()
    wc_img.to_image().save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img = ImageReader(img_buffer)
    c.drawImage(img, 50, height-400, width=500, height=200)
    y_pos = height-430
    c.setFont("Helvetica", 12)
    for idx, row in df.iterrows():
        if y_pos < 50:
            c.showPage()
            y_pos = height-50
        text = f"{idx+1}. [{row['language']}] {row['text']} | Sentiment: {row['sentiment']} | Emotion: {row['top_emotion']} | Summary: {row['summary']} | Topic: {row['topic']}"
        c.drawString(50, y_pos, text)
        y_pos -= 20
    c.save()
    buffer.seek(0)
    return buffer

def cluster_topics(texts, n_topics=5):
    if len(texts) < n_topics: n_topics = len(texts)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_topics, random_state=42)
    kmeans.fit(X)
    clusters = kmeans.labels_
    return clusters

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="🌍 e-Consultation Dashboard", layout="wide")
st.title("🌍 Advanced e-Consultation Analytics Platform")

if "all_results" not in st.session_state: st.session_state.all_results = []

tab1, tab2, tab3 = st.tabs(["📝 Text","🎤 Audio","🎬 Video"])

with tab1:
    comments = st.text_area("Enter comments (one per line)", height=200)
    if st.button("Analyze Text", key="text"):
        if comments.strip():
            for line in comments.split("\n"):
                sentiment, s_score, lang = analyze_sentiment(line)
                summary = summarize_text(line)
                emotion, e_score, all_emotions = analyze_emotion(line)
                st.session_state.all_results.append({
                    'text': line, 'sentiment': sentiment, 'summary': summary,
                    'top_emotion': emotion, 'emotion_score': e_score, 'all_emotions': all_emotions,
                    'language': lang
                })

with tab2:
    audio_file = st.file_uploader("Upload audio file", type=["wav","mp3"])
    if audio_file and st.button("Analyze Audio", key="audio"):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp.write(audio_file.read())
        temp.close()
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp.name) as source:
            audio_data = recognizer.record(source)
            try: text = recognizer.recognize_google(audio_data)
            except: text = ""
        sentiment, s_score, lang = analyze_sentiment(text)
        summary = summarize_text(text)
        emotion, e_score, all_emotions = analyze_emotion(text)
        st.session_state.all_results.append({
            'text': text, 'sentiment': sentiment, 'summary': summary,
            'top_emotion': emotion, 'emotion_score': e_score, 'all_emotions': all_emotions,
            'language': lang
        })

with tab3:
    video_file = st.file_uploader("Upload video file", type=["mp4","mov"])
    if video_file and st.button("Analyze Video", key="video"):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(video_file.read())
        temp.close()
        clip = mp.VideoFileClip(temp.name)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        clip.audio.write_audiofile(temp_audio.name)
        clip.close()
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio.name) as source:
            audio_data = recognizer.record(source)
            try: text = recognizer.recognize_google(audio_data)
            except: text = ""
        sentiment, s_score, lang = analyze_sentiment(text)
        summary = summarize_text(text)
        emotion, e_score, all_emotions = analyze_emotion(text)
        st.session_state.all_results.append({
            'text': text, 'sentiment': sentiment, 'summary': summary,
            'top_emotion': emotion, 'emotion_score': e_score, 'all_emotions': all_emotions,
            'language': lang
        })

# ---------------- DASHBOARD ----------------
if st.session_state.all_results:
    df = pd.DataFrame(st.session_state.all_results)

    # KPI CARDS
    st.subheader("📊 Dashboard Metrics")
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div style='background-color:#B2DFDB; padding:15px; border-radius:15px; text-align:center'><h3>📝 Total Feedback</h3><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div style='background-color:#FFCCBC; padding:15px; border-radius:15px; text-align:center'><h3>😊 Top Emotion</h3><h2>{df['top_emotion'].mode()[0]}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div style='background-color:#C5CAE9; padding:15px; border-radius:15px; text-align:center'><h3>🌐 Most Active Language</h3><h2>{df['language'].mode()[0]}</h2></div>", unsafe_allow_html=True)

    # Language Distribution
    st.subheader("🌐 Language Distribution")
    lang_counts = df['language'].value_counts()
    fig_lang = px.pie(names=lang_counts.index, values=lang_counts.values, title="Feedback Language Distribution", color_discrete_sequence=pastel_colors)
    st.plotly_chart(fig_lang)

    # Sentiment per Language
    st.subheader("📊 Sentiment per Language")
    for lang in df['language'].unique():
        lang_df = df[df['language']==lang]
        fig = px.pie(lang_df, names='sentiment', title=f'Sentiment Distribution for {lang}', color_discrete_sequence=pastel_colors)
        st.plotly_chart(fig)

    # WordCloud
    st.subheader("☁️ WordCloud of Feedback")
    wc = generate_wordcloud(df['text'].tolist())

    # Topic Clustering
    st.subheader("🔑 Topic Clustering of Feedback")
    df['topic'] = cluster_topics(df['text'].tolist(), n_topics=5)
    for topic_num in sorted(df['topic'].unique()):
        topic_df = df[df['topic']==topic_num]
        with st.expander(f"Topic {topic_num+1} ({len(topic_df)} feedback items)"):
            for idx, row in topic_df.iterrows():
                st.markdown(f"<div style='background-color:#FFF3E0; padding:10px; border-radius:10px'><b>Language:</b> {row['language']}<br><b>Text:</b> {row['text']}<br><b>Sentiment:</b> {row['sentiment']}<br><b>Top Emotion:</b> {row['top_emotion']}<br><b>Summary:</b> {row['summary']}</div>", unsafe_allow_html=True)

    # WordCloud per Topic
    st.subheader("☁️ WordCloud per Topic")
    for topic_num in sorted(df['topic'].unique()):
        topic_texts = df[df['topic']==topic_num]['text'].tolist()
        st.markdown(f"**Topic {topic_num+1}**")
        generate_wordcloud(topic_texts)

    # PDF Download
    st.subheader("💾 Download PDF Report")
    pdf_buffer = generate_pdf(df, wc)
    st.download_button(label="Download PDF", data=pdf_buffer, file_name="econsultation_report.pdf", mime="application/pdf")

    # CSV Download
    st.subheader("💾 Download CSV Report")
    csv_buffer = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download CSV", data=csv_buffer, file_name="econsultation_report.csv", mime="text/csv")
