import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import re
from collections import Counter

# ===============================
# Load ML Pipeline + Dataset
# ===============================
@st.cache_resource
def load_pipeline():
    vec = joblib.load("pktfidf_vectorizer.pkl")
    model = joblib.load("pksvm_model.pkl")
    return vec, model

@st.cache_data
def load_dataset():
    df = pd.read_csv("dataset_75k_trend_built.csv")
    return df

vec, model = load_pipeline()
sample_df = load_dataset()

# ===============================
# Custom CSS
# ===============================
# ===============================
# Custom CSS
# ===============================
st.markdown("""
    <style>
    /* Background gradient */
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        font-family: 'Segoe UI', sans-serif;
        color: #f8f9fa;
    }

    /* Title styles */
    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 46px;
        font-weight: 900;
        margin-bottom: 5px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        letter-spacing: 1px;
    }
    .sub-title {
        text-align: center;
        color: #dcdcdc;
        font-size: 20px;
        margin-bottom: 30px;
        font-style: italic;
    }

    /* Text input box */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #00c6ff;
        padding: 12px;
        font-size: 16px;
        background-color: #1e1e2f;
        color: white;
    }
    .stTextArea textarea:focus {
        border-color: #ff6a00;
        box-shadow: 0 0 10px #ff6a00;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #ff512f, #dd2476);
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.3);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2f;
        color: #ffffff;
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #0072ff;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff512f, #dd2476) !important;
        color: white !important;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }

    /* Dataframe table */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid #0072ff;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #dcdcdc;
        font-size: 14px;
        opacity: 0.7;
    }
    
    </style>
""", unsafe_allow_html=True)


# ===============================
# App Title
# ===============================
st.markdown('<div class="main-title">India Sentiment Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Classify a post/tweet as Pro-India, Anti-India, or Neutral</div>', unsafe_allow_html=True)

# ===============================
# Session State for User History
# ===============================
if "history" not in st.session_state:
    st.session_state["history"] = []

# ===============================
# Helpers
# ===============================
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # keep only English words
    return text.lower()

def predict_sentiment(text):
    X = vec.transform([text])
    pred = model.predict(X)[0]
    return pred

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["🔮 Classify Text", "📊 Dashboard", "📂 Dataset Preview"])

# -------- TAB 1: Classify --------
with tab1:
    user_input = st.text_area("Enter your content here:", "")

    if st.button("🔮 Classify Sentiment"):
        if user_input.strip():
            sentiment = predict_sentiment(user_input)
            st.session_state["history"].append(sentiment)

            emoji_map = {
                "Pro-India": "Pro-India",
                "Anti-India": "Anti-India",
                "Neutral": "Neutral"
            }

            st.subheader("Prediction:")
            st.markdown(f"<h2 style='text-align:center; color:white;'>{emoji_map.get(sentiment, '🤔 Unknown')}</h2>", unsafe_allow_html=True)

            # Word frequency chart for current input
            cleaned = clean_text(user_input)
            words = cleaned.split()
            word_counts = Counter(words)
            if word_counts:
                df = [{"Word": k, "Count": v} for k, v in word_counts.items()]
                fig = px.bar(df, x="Word", y="Count", color="Count",
                             color_continuous_scale="sunset", title="Most Used Words")
                fig.update_layout(
                    title_x=0.5,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white")
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Please enter some text.")

# -------- TAB 2: Dashboard --------
with tab2:
    st.markdown("## Sentiment Dashboard (from Dataset)")

    # Ensure dataset has Sentiment column
    if "sentiment" in sample_df.columns:
        dataset_counts = sample_df["sentiment"].value_counts().reset_index()
        dataset_counts.columns = ["Sentiment", "Count"]

        pie_fig = px.pie(dataset_counts, names="Sentiment", values="Count", hole=0.4,
                         color="Sentiment",
                         color_discrete_map={"Pro-India": "green",
                                             "Anti-India": "red",
                                             "Neutral": "gray"},
                         title="Sentiment Distribution in Sample Dataset")
        pie_fig.update_layout(title_x=0.5)
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.error("The dataset does not contain a 'Sentiment' column.")
    if "sentiment" in sample_df.columns and "hashtags" in sample_df.columns:
        anti_df = sample_df[sample_df["sentiment"] == "Anti-India"]

        hashtags = []
        for txt in anti_df["hashtags"].dropna().astype(str):
            tags = re.findall(r"#\w+", txt)  # extract hashtags
            hashtags.extend(tags)

        if hashtags:
            hashtag_counts = Counter([tag.lower() for tag in hashtags])
            top_hashtags = hashtag_counts.most_common(15)  # top 15

            hash_df = pd.DataFrame(top_hashtags, columns=["Hashtag", "Count"])

            st.markdown("**Most Used Hashtags in Anti-India Posts**")
            bar_fig = px.bar(
                hash_df,
                x="Hashtag",
                y="Count",
                color="Count",
                color_continuous_scale="Reds",
                title="Top Anti-India Hashtags"
            )
            bar_fig.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                title_x=0.5
            )
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.info("No hashtags found in Anti-India posts.")
# -------- TAB 3: Dataset Preview --------
with tab3:
    st.markdown("Sample Dataset (first 10 rows)")
    st.dataframe(sample_df.head(10))

# ===============================
# Footer
# ===============================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#f0f0f0;'>✨ Built with Streamlit, Plotly, and SVM</p>", unsafe_allow_html=True)
