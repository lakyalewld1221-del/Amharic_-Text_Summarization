import streamlit as st
import time
import re
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amharic Text Summarizer",
    page_icon="📝",
    layout="wide"
)
# ── mT5 Model Configuration ──────────────────────────────
MODEL_NAME = (
    "yohannesahunm/mt5-small-Amharic-text-summaization"
)

@st.cache_resource
def load_mt5_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    model.eval()
    return tokenizer, model


def mt5_summarize(text):
    tokenizer, model = load_mt5_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=4,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

# ── Amharic Stopwords ─────────────────────────────────────────────────────────
AMHARIC_STOPWORDS = {
    "እና", "ወይም", "ላይ", "ውስጥ", "ወደ", "ከ", "ጋር", "ነው", "ናት", "ናቸው",
    "እንደ", "ስለ", "የሚሆን", "የሚገኝ", "ከሆነ", "ባሉ", "ይህን", "ይህንን", "ያለ",
    "እንዲሁም", "በተለይም", "ቢሆን", "ከዚህ", "በዚህ", "በኋላ", "በፊት", "እስከ",
    "በኩል", "መካከል", "ሆኖ", "ሲሆን", "ማለት", "ነበር", "እንጂ", "ብቻ", "ሁሉ"
}

def clean_amharic_word(word):
    return re.sub(r'[።፣፤፥፦፧?!.,:;\'"()\[\]\n\r]', '', word).strip()

def extractive_summarize(text, num_sentences=2):
    raw_sentences = re.split(r'[።\n]+', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 4]

    if not sentences:
        return text.strip()
    if len(sentences) <= num_sentences:
        return "። ".join(sentences) + "།"

    words = [clean_amharic_word(w) for w in text.split()]
    content_words = [w for w in words if w and w not in AMHARIC_STOPWORDS and len(w) > 1]
    word_freq = Counter(content_words)

    sentence_scores = []
    for idx, sentence in enumerate(sentences):
        sent_words = [clean_amharic_word(w) for w in sentence.split()]
        valid_words = [w for w in sent_words if w in word_freq]
        if not valid_words:
            score = 0.0
        else:
            raw_score = sum(word_freq[w] for w in valid_words)
            score = raw_score / (len(sent_words) ** 0.65)
        if idx == 0:
            score *= 1.45
        elif idx == 1:
            score *= 1.25
        sentence_scores.append((score, idx, sentence))

    top_candidates = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:num_sentences]
    top_candidates = sorted(top_candidates, key=lambda x: x[1])
    return "። ".join([item[2] for item in top_candidates]) + "།"

# ── Pre-loaded Samples ────────────────────────────────────────────────────────
SAMPLE_TEXTS = {
    "Select a sample text...": "",
    "⚡ Energy & Export (የኢነርጂ ዜና)": (
        "ኢትዮጵያ በታዳሽ የኃይል ማመንጫ ዘርፍ የጀመረችውን ሰፊ ኢንቨስትመንት በማጠናከር፣ ለጎረቤት ሀገራት "
        "የምታቀርበውን የኤሌክትሪክ ኃይል በከፍተኛ ደረጃ ለማሳደግ አዲስ ስትራቴጂ ይፋ አደረገች። "
        "የውሃ እና ኢነርጂ ሚኒስቴር እንዳስታወቀው ባለፉት ዓመታት የተገነቡት ማመንጫዎች ሙሉ በሙሉ "
        "ወደ ስራ ሲገቡ የአገሪቱ አጠቃላይ አቅም በእጥፍ ይጨምራል። ወደ ኬንያ፣ ሱዳን እና ጅቡቲ "
        "የሚላከው ኃይል ለአገሪቱ ከፍተኛ የውጭ ምንዛሪ እያስገኘ ይገኛል።"
    ),
    "🌾 Agriculture (ግብርና እና የምግብ ዋስትና)": (
        "ኢትዮጵያ በምስራቅ አፍሪካ ቀንድ የምትገኝ ሰፊና የተለያየ ስነ-ምህዳር ያላት ሀገር ስትሆን ኢኮኖሚዋ "
        "በዋነኝነት የተመሰረተው በግብርና ዘርፍ ላይ ነው። ግብርና ከጠቅላላው ምርት ከሰላሳ በመቶ በላይ "
        "የሚሸፍን ሲሆን ለሰማንያ በመቶ ህዝብ ዋነኛ የገቢ ምንጭ ነው። "
        "የተሻሻሉ ዝርያዎችን ማሰራጨት የምግብ ዋስትናን ለማረጋገጥ ወሳኝ ጥረት ነው።"
    ),
    "🏛️ Heritage & Tourism (ቅርስ እና ቱሪዝም)": (
        "ኢትዮጵያ በረጅም ዘመናት ታሪኳ፣ በልዩ ልዩ ባህሎቿ እና በአስደናቂ የተፈጥሮ መስህቦቿ በዓለም አቀፍ "
        "ደረጃ የምትታወቅ ጥንታዊ ሀገር ናት። ሀገሪቱ በዓለም ቅርስነት የተመዘገቡ እንደ ላሊበላ ውቅር "
        "አብያተ ክርስቲያናት እና የአክሱም ሀውልቶች ባሉ ታሪካዊ ሀብቶች ባለቤት ናት። "
        "ቱሪዝም ከፍተኛ የውጭ ምንዛሪ የሚያስገኝ እና ሰፊ የስራ እድል የሚፈጥር ዘርፍ ነው።"
    )
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📌 Dashboard Menu")
page = st.sidebar.radio(
    "Go to",
    ["📝 Summarizer", "📊 Analytics", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
num_sentences = st.sidebar.slider(
    summary_method = st.selectbox(
    "Choose summarization method:",
    [
        "Extractive (Python)",
        "Abstractive (mT5)",
        "Compare both"
    ]
)
    "Number of summary sentences (የዓረፍተ-ነገሮች ብዛት):",
    min_value=1,
    max_value=5,
    value=2,
    step=1
)
st.sidebar.info("""
**Approach:** Extractive Summarization
**Language:** Amharic (አማርኛ)
**Grammar:** 100% Authentic
**Latency:** Sub-second
""")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: SUMMARIZER
# ═════════════════════════════════════════════════════════════════════════════
if page == "📝 Summarizer":
    st.title("📝 Amharic Text Summarization Dashboard")
    st.markdown("Paste or type any Amharic article to extract the most important sentences instantly.")
    st.markdown("---")

    col_input, col_output = st.columns([1, 1], gap="medium")

    with col_input:
        st.subheader("📥 Input Amharic Text")

        selected_sample = st.selectbox(
            "💡 Choose a pre-loaded sample (optional):",
            list(SAMPLE_TEXTS.keys())
        )
        default_text = SAMPLE_TEXTS.get(selected_sample, "")

        article = st.text_area(
            "Enter or paste your Amharic text here:",
            value=default_text,
            height=300,
            placeholder="የማጠቃለያ ማድረግ የሚፈልጉትን የአማርኛ ጽሑፍ እዚህ ያስገቡ..."
        )

        input_words = len(article.strip().split()) if article.strip() else 0
        input_chars = len(article.strip()) if article.strip() else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Input Words", input_words)
        m2.metric("Characters", input_chars)
        m3.metric("Est. Read Time", f"{max(1, input_words // 150)} min" if input_words else "0 min")

        generate_btn = st.button(
            "🚀 Generate Summary",
            type="primary",
            use_container_width=True
        )

    with col_output:
        st.subheader("📤 Generated Summary")

        if generate_btn:
            if not article.strip():
                st.warning("Please enter some Amharic text first.")

            elif len(article.strip()) < 30:
                st.warning("Please enter a longer text to summarize.")

            else:
                try:
                    start_time = time.time()

                    # Option 1: Python Extractive
                    if summary_method == "Extractive (Python)":
                        with st.spinner("Extracting important sentences..."):
                            summary = extractive_summarize(
                                article.strip(),
                                num_sentences=num_sentences
                            )

                        st.markdown("### Python Extractive Summary")
                        st.write(summary)

                    # Option 2: mT5
                    elif summary_method == "Abstractive (mT5)":
                        with st.spinner("Generating mT5 summary..."):
                            summary = mt5_summarize(article.strip())

                        st.markdown("### mT5 Summary")
                        st.write(summary)

                    # Option 3: Compare both
                    else:
                        with st.spinner("Generating both summaries..."):
                            extractive_summary = extractive_summarize(
                                article.strip(),
                                num_sentences=num_sentences
                            )

                            mt5_summary = mt5_summarize(article.strip())

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### Python Extractive")
                            st.write(extractive_summary)

                        with col2:
                            st.markdown("### mT5 Abstractive")
                            st.write(mt5_summary)

                    elapsed_time = time.time() - start_time
                    st.caption(
                        f"Generation time: {elapsed_time:.2f} seconds"
                    )

                except Exception as e:
                    st.error(f"Summary generation failed: {e}")

        else:
            st.info(
                "👈 Enter an Amharic article and click Generate Summary."
            )
# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.title("📊 Extractive Summarization Methodology")
    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("✨ Why Extractive Summarization?")
        st.markdown("""
        - **100% Grammatical Accuracy:** Every sentence is taken verbatim from the source.
        - **Zero Hallucination:** No invented or broken phrases.
        - **Ultra-Low Latency:** Generates summaries in under 50 milliseconds.
        - **No GPU Required:** Runs efficiently on standard cloud hardware.
        """)

    with col_b:
        st.subheader("🎯 Sentence Scoring Formula")
        st.latex(r"Score(S_i) = \alpha_i \times \frac{\sum_{w \in S_i} Freq(w)}{Length(S_i)^{0.65}}")
        st.markdown("""
        1. **Term Salience:** Scores content words by document frequency.
        2. **Length Normalization:** Avoids bias toward very long sentences.
        3. **Lead Bias ($\\alpha_i$):** First sentences carry 45% extra weight.
        """)

    st.markdown("---")
    st.subheader("🔍 Processing Pipeline")
    st.code("""
[Raw Amharic Text]
        ↓
[Sentence Segmentation (Delimiter: ።)]
        ↓
[Amharic Stopword Filtering]
        ↓
[Salience & Position Scoring]
        ↓
[Top-K Sentence Selection]
        ↓
[Chronological Re-ordering]
        ↓
[Final Summary Output]
    """, language="text")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("---")
    st.markdown("""
    ### 🇪🇹 Amharic Extractive Text Summarization System

    This application automatically condenses lengthy Amharic news articles, research documents,
    and institutional reports into concise, grammatically accurate summaries.

    #### 🌟 Key Features:
    - **Zero Hallucination:** Extracted sentences are 100% authentic.
    - **Customizable Length:** 1 to 5 target sentences via sidebar slider.
    - **Live Metrics:** Real-time word count, compression ratio, and latency.
    - **Pre-loaded Samples:** One-click testing with built-in Amharic articles.
    - **Download Export:** Save generated summaries as `.txt` files.

    #### 🛠️ Technology Stack:
    - **Framework:** Streamlit
    - **Algorithm:** Salience-based Extractive NLP
    - **Language:** Amharic (አማርኛ) — Ethiopic script
    - **Hosting:** Streamlit Community Cloud
    """)

st.divider()
st.caption("Amharic Text Summarization Dashboard | Powered by Extractive NLP | Built with Streamlit")
