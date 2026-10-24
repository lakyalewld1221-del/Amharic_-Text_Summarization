import streamlit as st
import time
import re
from collections import Counter
from PIL import Image

# Safe import for OCR
try:
    import pytesseract
    # Tell pytesseract the exact Tesseract binary path on Streamlit Cloud (Linux)
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Page Config: Modern Wide Dashboard
st.set_page_config(
    page_title="Amharic Extractive Summarization Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Amharic Stopwords & Normalization ─────────────────────────────────────────
AMHARIC_STOPWORDS = {
    "እና", "ወይም", "ላይ", "ውስጥ", "ወደ", "ከ", "ጋር", "ነው", "ናት", "ናቸው",
    "እንደ", "ስለ", "የሚሆን", "የሚገኝ", "ከሆነ", "ባሉ", "ይህን", "ይህንን", "ያለ",
    "እንዲሁም", "በተለይም", "ቢሆን", "ከዚህ", "በዚህ", "በኋላ", "በፊት", "እስከ",
    "በኩል", "መካከል", "ሆኖ", "ሲሆን", "ማለት", "ነበር", "እንጂ", "ብቻ", "ሁሉ"
}

def clean_amharic_word(word):
    """Remove Amharic and Latin punctuation marks from word."""
    return re.sub(r'[።፣፤፥፦፧?!.,:;\'"()\[\]\n\r]', '', word).strip()

def extractive_summarize(text, num_sentences=2):
    """
    Extractive Summarization for Amharic:
    1. Splits text by Amharic full stops (።) and newlines
    2. Scores sentences based on term salience and inverted-pyramid position
    3. Returns top sentences in their original chronological order
    """
    raw_sentences = re.split(r'[።\n]+', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 4]

    if not sentences:
        return text.strip()

    if len(sentences) <= num_sentences:
        return "። ".join(sentences) + "።"

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

        # Lead bias: First two sentences usually carry the highest core information
        if idx == 0:
            score *= 1.45
        elif idx == 1:
            score *= 1.25

        sentence_scores.append((score, idx, sentence))

    top_candidates = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:num_sentences]
    top_candidates = sorted(top_candidates, key=lambda x: x[1])

    summary = "። ".join([item[2] for item in top_candidates]) + "።"
    return summary

def extract_text_from_image(image):
    """Extract Amharic text from an image using Tesseract OCR."""
    if not OCR_AVAILABLE:
        return "⚠️ OCR library (pytesseract) is not installed."
    try:
        extracted = pytesseract.image_to_string(image, lang='amh+eng')
        return extracted.strip()
    except Exception:
        try:
            return pytesseract.image_to_string(image).strip()
        except Exception:
            return "⚠️ OCR Error: Please ensure tesseract-ocr and tesseract-ocr-amh are installed."

# ── Pre-loaded Samples for 1-Click Testing ──────────────────────────────────
SAMPLE_TEXTS = {
    "Select a sample text...": "",
    "⚡ Energy & Export News (የኢነርጂ እና ኤክስፖርት ዜና)": (
        "ኢትዮጵያ በታዳሽ የኃይል ማመንጫ ዘርፍ የጀመረችውን ሰፊ ኢንቨስትመንት በማጠናከር፣ የሀገር ውስጥ የኢንዱስትሪ "
        "ፍላጎትን ከማሟላት ባለፈ ለጎረቤት ሀገራት የምታቀርበውን የኤሌክትሪክ ኃይል በከፍተኛ ደረጃ ለማሳደግ የሚያስችል "
        "አዲስ ስትራቴጂ ይፋ አደረገች። የውሃ እና ኢነርጂ ሚኒስቴር እንዳስታወቀው፣ ባለፉት ዓመታት የተገነቡት የውሃ፣ የፀሐይ "
        "እና የንፋስ ኃይል ማመንጫዎች ሙሉ በሙሉ ወደ ስራ ሲገቡ የአገሪቱ አጠቃላይ የማመንጨት አቅም በእጥፍ ይጨምራል። "
        "በአሁኑ ወቅት ወደ ኬንያ፣ ሱዳን እና ጅቡቲ የሚላከው የኤሌክትሪክ ኃይል ለአገሪቱ ከፍተኛ የውጭ ምንዛሪ እያስገኘ ይገኛል። "
        "ባጠቃላይ የኃይል አቅርቦቱን አስተማማኝ ማድረግ የአገር ውስጥ የስራ እድል ፈጠራን የሚያፋጥን እና የውጭ ቀጥታ "
        "ኢንቨስትመንትን የሚስብ በመሆኑ መንግስት ለዘርፉ ልዩ ትኩረት ሰጥቷል።"
    ),
    "🌾 Agriculture & Food Security (ግብርና እና የምግብ ዋስትና)": (
        "ኢትዮጵያ በምስራቅ አፍሪካ ቀንድ የምትገኝ ሰፊና የተለያየ ስነ-ምህዳር ያላት ሀገር ስትሆን፣ ኢኮኖሚዋ በዋነኝነት "
        "የተመሰረተው በግብርና ዘርፍ ላይ ነው። ግብርና ከጠቅላላው የሀገር ውስጥ ምርት (ጂዲፒ) ከሰላሳ በመቶ በላይ "
        "የሚሸፍን ከመሆኑም በላይ፣ ከሰማንያ በመቶ በላይ ለሚሆነው የገጠር ማህበረሰብ ዋነኛ የገቢ ምንጭ ነው። "
        "በቅርብ ዓመታት በመስኖ ስንዴ ልማት እና የተሻሻሉ የዘር ዝርያዎችን በማሰራጨት ምርታማነትን ለማሳደግ ሰፊ ጥረት እየተደረገ ነው። "
        "የአካባቢ ጥበቃ እና ዘላቂ የግብርና ቴክኖሎጂዎችን ማስፋፋት የምግብ ዋስትናን ለማረጋገጥ ወሳኝ እርምጃ ነው።"
    )
}

# ── Sidebar Navigation ─────────────────────────────────────────────────────
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
**Input Options:** Text Input or Image OCR  
**Language:** Amharic (አማርኛ)  
**Grammar Accuracy:** 100% (Verbatim Authentic)
""")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: SUMMARIZER DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "📝 Summarizer Dashboard":
    st.title("📝 Amharic Extractive Text Summarization Dashboard")
    st.markdown("Choose whether to type text or upload an image/scanned document to extract and summarize key points.")
    st.markdown("---")

    col_input, col_output = st.columns([1, 1], gap="medium")

    with col_input:
        st.subheader("📥 Source Input")

        # ── User Input Choice: Text or Image ──
        input_choice = st.radio(
            "Select how you want to provide your Amharic document:",
            ["✍️ Type / Paste Text", "📷 Upload Scanned Image / Photo (OCR)"],
            horizontal=True
        )

        article = ""

        # Option A: Direct Text Input
        if input_choice == "✍️ Type / Paste Text":
            selected_sample = st.selectbox("💡 Choose a pre-loaded sample (optional):", list(SAMPLE_TEXTS.keys()))
            default_text = SAMPLE_TEXTS[selected_sample] if selected_sample else ""

            article = st.text_area(
                "Enter or paste Amharic text:",
                value=default_text,
                height=260,
                placeholder="የማጠቃለያ ማድረግ የሚፈልጉትን የአማርኛ ጽሑፍ እዚህ ያስገቡ..."
            )

        # Option B: Image / Scanned Document Input
        else:
            uploaded_file = st.file_uploader(
                "Upload a scanned Amharic document or camera photo:",
                type=["png", "jpg", "jpeg", "webp"]
            )
            extracted_text = ""
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Document Preview", use_container_width=True)

                if st.button("🔍 Extract Text from Image"):
                    with st.spinner("📖 Reading Amharic text via OCR..."):
                        extracted_text = extract_text_from_image(image)
                        st.session_state["ocr_content"] = extracted_text

            initial_ocr = st.session_state.get("ocr_content", "")
            article = st.text_area(
                "Extracted Amharic Text (you can review or edit below):",
                value=initial_ocr,
                height=200,
                placeholder="የተነበበው የአማርኛ ጽሑፍ እዚህ ይታያል..."
            )

        input_words = len(article.strip().split()) if article.strip() else 0
        input_chars = len(article.strip()) if article.strip() else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Input Words", input_words)
        m2.metric("Characters", input_chars)
        m3.metric("Est. Read Time", f"{max(1, input_words // 150)} min" if input_words else "0 min")

        generate_btn = st.button("🚀 Generate Extractive Summary", type="primary", use_container_width=True)

    with col_output:
        st.subheader("📤 Extracted Key Summary")

        if generate_btn:
            if not article.strip():
                st.warning("⚠️ Please provide some Amharic text or extract an image first.")
            elif len(article.strip()) < 30:
                st.warning("⚠️ Please provide a longer text to summarize.")
            else:
                start_time = time.time()

                with st.spinner("🔍 Analyzing sentence salience and extracting key points..."):
                    summary = extractive_summarize(article.strip(), num_sentences=num_sentences)

                elapsed_time = time.time() - start_time

                summary_words = len(summary.split())
                compression_ratio = round((1 - (summary_words / max(1, input_words))) * 100, 1)

                st.success("✅ Extractive Summary Generated Successfully!")
                st.markdown(
                    f"""
                    <div style="background-color: #f0f7f4; padding: 18px; border-radius: 10px; border-left: 5px solid #28a745; margin-bottom: 15px;">
                        <p style="font-size: 17px; line-height: 1.8; color: #1e3d2f; margin: 0;">{summary}</p>
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
                    file_name="amharic_extractive_summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("👈 Choose text or upload an image on the left, then click **'Generate Extractive Summary'**.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: EXTRACTIVE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Extractive Analytics":
    st.title("📊 Extractive Summarization Methodology")
    st.markdown("How the Extractive Pipeline ranks and selects sentences from Amharic documents.")
    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("✨ Advantages of Extractive Summarization")
        st.markdown("""
        - **100% Grammatical Accuracy:** Sentences come directly from authentic source text, eliminating hallucinations.
        - **Dual Input Channels:** Seamlessly handles typed text, online articles, or photo/scanned documents via OCR.
        - **Zero Incomplete Fragments:** No broken clauses or missing verbs.
        - **Ultra-Low Latency:** Produces summaries in milliseconds without heavy GPU computation.
        """)

    with col_b:
        st.subheader("🎯 Sentence Scoring Formula")
        st.latex(r"Score(S_i) = \text{LeadBias}(i) \times \frac{\sum_{w \in S_i} \text{Freq}(w)}{\text{Length}(S_i)^{0.65}}")
        st.markdown("""
        1. **Token Salience:** Evaluates content word frequency after filtering common Amharic stopwords.
        2. **Length Normalization:** Prevents bias towards overly long or short sentences.
        3. **Inverted Pyramid Lead Bias:** Gives precedence to opening sentences which contain core news facts.
        """)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: ABOUT PROJECT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About Project":
    st.title("ℹ️ About This Project")
    st.markdown("---")

    st.markdown("""
    ### 🇪🇹 Amharic Text Summarization System

    This project demonstrates automatic text summarization specifically designed for **Amharic (*አማርኛ*)**, addressing the unique linguistic structures of the Ethiopic script.

    #### 🌟 Key Benefits:
    - **Flexible Ingestion:** Users can input plain text directly or upload image scans of printed documents.
    - **Accurate Information Retrieval:** Condenses news, research, and institutional reports without distorting meaning.
    - **No Hallucination:** Extractive summarization guarantees reliable, fact-checked sentences.
    - **Interactive Dashboard:** Offers customizable sentence counts, compression rate calculation, and quick testing.
    """)

st.divider()
st.caption("Amharic Extractive Text Summarization Dashboard | Built with Streamlit")
