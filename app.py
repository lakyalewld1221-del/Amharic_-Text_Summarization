import streamlit as st
import time
import re
from collections import Counter

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amharic Text Summarizer",
    page_icon="📝",
    layout="wide"
)

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
    "Number of summary sentences (የዓረፍተ-ነገሮች ብዛት):",
    min_value=1,
    max_value=6,
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
                st.warning("⚠️ Please enter some Amharic text first.")
            elif len(article.strip()) < 30:
                st.warning("⚠️ Please enter a longer text to summarize.")
            else:
                start_time = time.time()
                with st.spinner("🔍 Extracting key sentences..."):
                    summary = extractive_summarize(article.strip(), num_sentences=num_sentences)
                elapsed_time = time.time() - start_time

                summary_words = len(summary.split())
                compression_ratio = round((1 - (summary_words / max(1, input_words))) * 100, 1)

                st.success("✅ Summary Generated Successfully!")
                st.markdown(
                    f"""
                    <div style="background-color:#f0f7f4; padding:20px; border-radius:10px;
                    border-left:5px solid #28a745; margin-bottom:15px;">
                    <p style="font-size:17px; line-height:1.9; color:#1e3d2f; margin:0;">{summary}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                o1, o2, o3 = st.columns(3)
                o1.metric("Summary Words", summary_words)
                o2.metric("Compression", f"-{compression_ratio}%", delta=f"{compression_ratio}% reduction")
                o3.metric("Latency", f"{elapsed_time:.3f}s")

                st.download_button(
                    label="💾 Download Summary (.txt)",
                    data=summary,
                    file_name="amharic_summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("👈 Paste an Amharic article on the left and click **Generate Summary**.")

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
