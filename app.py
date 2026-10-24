import streamlit as st
import time
import re
from collections import Counter
from PIL import Image

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amharic Extractive Summarization Dashboard",
    page_icon="📊",
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
    return "། ".join([item[2] for item in top_candidates]) + "།"

# ── Pre-loaded Samples ────────────────────────────────────────────────────────
SAMPLE_TEXTS = {
    "Select a sample text...": "",
    "⚡ Energy & Export News (የኢነርጂ ዜና)": (
        "ኢትዮጵያ በታዳሽ የኃይል ማመንጫ ዘርፍ የጀመረችውን ሰፊ ኢንቨስትመንት በማጠናከር፣ ለጎረቤት ሀገራት "
        "የምታቀርበውን የኤሌክትሪክ ኃይል በከፍተኛ ደረጃ ለማሳደግ አዲስ ስትራቴጂ ይፋ አደረገች። "
        "የውሃ እና ኢነርጂ ሚኒስቴር እንዳስታወቀው፣ ባለፉት ዓመታት የተገነቡት ማመንጫዎች ሙሉ በሙሉ "
        "ወደ ስራ ሲገቡ የአገሪቱ አጠቃላይ አቅም በእጥፍ ይጨምራል። ወደ ኬንያ፣ ሱዳን እና ጅቡቲ "
        "የሚላከው ኃይል ለአገሪቱ ከፍተኛ የውጭ ምንዛሪ እያስገኘ ይገኛል።"
    ),
    "🌾 Agriculture & Food Security (ግብርና)": (
        "ኢትዮጵያ በምስራቅ አፍሪካ ቀንድ የምትገኝ ሰፊና የተለያየ ስነ-ምህዳር ያላት ሀገር ስትሆን ኢኮኖሚዋ "
        "በዋነኝነት የተመሰረተው በግብርና ዘርፍ ላይ ነው። ግብርና ከጠቅላላው ምርት ከሰላሳ በመቶ በላይ "
        "የሚሸፍን ሲሆን ለሰማንያ በመቶ ህዝብ ዋነኛ የገቢ ምንጭ ነው። "
        "የተሻሻሉ ዝርያዎችን ማሰራጨት የምግብ ዋስትናን ለማረጋገጥ ወሳኝ ጥረት ነው።"
    )
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📌 Dashboard Menu")
page = st.sidebar.radio(
    "Go to",
    ["📝 Summarizer Dashboard", "📊 Extractive Analytics", "ℹ️ About Project"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Summary Settings")
num_sentences = st.sidebar.slider(
    "Target Sentences (የዓረፍተ-ነገሮች ብዛት):",
    min_value=1,
    max_value=5,
    value=2,
    step=1
)

st.sidebar.info("""
**Approach:** Extractive Summarization
**Input Options:**
- ✍️ Type or paste text
- 📷 Upload image + type text
**Language:** Amharic (አማርኛ)
**Grammar Accuracy:** 100%
""")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: SUMMARIZER DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "📝 Summarizer Dashboard":
    st.title("📝 Amharic Extractive Text Summarization Dashboard")
    st.markdown("Type Amharic text directly **or** upload a scanned document image and type the text from it.")
    st.markdown("---")

    col_input, col_output = st.columns([1, 1], gap="medium")

    with col_input:
        st.subheader("📥 Source Input")

        # ── Input Method Choice ──
        input_choice = st.radio(
            "Select your input method:",
            ["✍️ Type / Paste Text", "📷 Upload Document Image + Type Text"],
            horizontal=True
        )

        article = ""

        # ── OPTION A: Direct Text ──────────────────────────────────────────
        if input_choice == "✍️ Type / Paste Text":
            selected_sample = st.selectbox(
                "💡 Or choose a pre-loaded sample:",
                list(SAMPLE_TEXTS.keys())
            )
            default_text = SAMPLE_TEXTS.get(selected_sample, "")

            article = st.text_area(
                "Enter or paste your Amharic text below:",
                value=default_text,
                height=270,
                placeholder="የማጠቃለያ ማድረግ የሚፈልጉትን የአማርኛ ጽሑፍ እዚህ ያስገቡ..."
            )

        # ── OPTION B: Image Upload + Manual Text Entry ─────────────────────
        else:
            st.info(
                "📌 **How to use:**\n"
                "1. Upload your scanned document or photo below.\n"
                "2. Look at the image preview on the right.\n"
                "3. Type the Amharic text from the image into the text box.\n"
                "4. Click **Generate Extractive Summary**!"
            )

            col_img, col_txt = st.columns([1, 1], gap="small")

            with col_img:
                uploaded_file = st.file_uploader(
                    "📂 Upload Amharic document image:",
                    type=["png", "jpg", "jpeg", "webp", "bmp"]
                )
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(
                        image,
                        caption="📄 Your Uploaded Document",
                        use_container_width=True
                    )
                    st.caption("👆 Read the text from this image and type it in the box on the right.")
                else:
                    st.markdown(
                        """
                        <div style="background:#f9fafb; border: 2px dashed #d1d5db;
                        border-radius:10px; padding:40px; text-align:center; color:#9ca3af;">
                        <div style="font-size:40px">📷</div>
                        <div style="margin-top:10px; font-size:14px">Upload a scanned document<br>or photo here</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with col_txt:
                article = st.text_area(
                    "✍️ Type the Amharic text from the image here:",
                    height=270,
                    placeholder="ምስሉ ላይ ያለውን የአማርኛ ጽሑፍ እዚህ ያስገቡ...\n\nEnter the Amharic text visible in your uploaded image..."
                )

        # ── Input Metrics ──
        input_words = len(article.strip().split()) if article.strip() else 0
        input_chars = len(article.strip()) if article.strip() else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Input Words", input_words)
        m2.metric("Characters", input_chars)
        m3.metric("Est. Read Time", f"{max(1, input_words // 150)} min" if input_words else "0 min")

        generate_btn = st.button(
            "🚀 Generate Extractive Summary",
            type="primary",
            use_container_width=True
        )

    # ── Output Column ──────────────────────────────────────────────────────
    with col_output:
        st.subheader("📤 Extracted Key Summary")

        if generate_btn:
            if not article.strip():
                st.warning("⚠️ Please enter some Amharic text first.")
            elif len(article.strip()) < 30:
                st.warning("⚠️ Please provide a longer text (at least a few sentences).")
            else:
                start_time = time.time()
                with st.spinner("🔍 Analyzing and extracting key sentences..."):
                    summary = extractive_summarize(article.strip(), num_sentences=num_sentences)
                elapsed_time = time.time() - start_time

                summary_words = len(summary.split())
                compression_ratio = round((1 - (summary_words / max(1, input_words))) * 100, 1)

                st.success("✅ Extractive Summary Generated Successfully!")
                st.markdown(
                    f"""
                    <div style="background-color:#f0f7f4; padding:18px; border-radius:10px;
                    border-left:5px solid #28a745; margin-bottom:15px;">
                    <p style="font-size:17px; line-height:1.8; color:#1e3d2f; margin:0;">{summary}</p>
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
            st.info("👈 Enter text or upload an image and type the text, then click **Generate Extractive Summary**.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: EXTRACTIVE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Extractive Analytics":
    st.title("📊 Extractive Summarization Methodology")
    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("✨ Advantages")
        st.markdown("""
        - **100% Grammatical Accuracy:** Verbatim authentic sentences from the source.
        - **Dual Input Modes:** Typed text or scanned image (manual entry).
        - **Zero Hallucination:** No invented words or broken fragments.
        - **Ultra-Low Latency:** Sub-second processing speed.
        - **Cloud Reliable:** No system-level dependencies required.
        """)

    with col_b:
        st.subheader("🎯 Scoring Formula")
        st.latex(r"Score(S_i) = \alpha_i \times \frac{\sum_{w \in S_i} Freq(w)}{Length(S_i)^{0.65}}")
        st.markdown("""
        1. **Term Salience:** Scores sentences by high-frequency content words.
        2. **Length Normalization:** Balances short vs. long sentence bias.
        3. **Lead Bias:** Opening sentences receive higher weight.
        """)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: ABOUT PROJECT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About Project":
    st.title("ℹ️ About This Project")
    st.markdown("---")
    st.markdown("""
    ### 🇪🇹 Amharic Extractive Text Summarization System

    This app condenses lengthy Amharic documents into precise, grammatically accurate summaries.
    Users can input text directly or upload scanned document images and type the visible text.

    #### 🛠️ Technology Stack:
    - **Framework:** Streamlit
    - **Image Handling:** Pillow (PIL)
    - **NLP Engine:** Custom salience-based extractive algorithm
    - **Language:** Amharic (አማርኛ) — Ethiopic script
    - **Hosting:** Streamlit Community Cloud
    """)

st.divider()
st.caption("Amharic Extractive Text Summarization Dashboard | Built with Streamlit")
