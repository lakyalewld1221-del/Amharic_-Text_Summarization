import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "yohannesahunm/mt5-small-Amharic-text-summaization"

st.set_page_config(
    page_title="Amharic Text Summarizer",
    page_icon="📝",
    layout="centered"
)

@st.cache_resource
def load_summarization_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

st.title("📝 Amharic Text Summarization")
st.write(
    "Paste an Amharic article or paragraph below "
    "to generate a shorter summary using AI."
)

with st.spinner("Loading the summarization model..."):
    tokenizer, model = load_summarization_model()

article = st.text_area(
    "Enter Amharic text",
    height=250,
    placeholder="የማጠቃለያ ማድረግ የሚፈልጉትን የአማርኛ ጽሑፍ እዚህ ያስገቡ..."
)

max_words = st.slider(
    "Maximum generated tokens",
    min_value=30,
    max_value=150,
    value=100,
    step=10
)

if st.button("Generate Summary", type="primary"):
    if not article.strip():
        st.warning("Please enter some Amharic text first.")
    elif len(article.strip()) < 30:
        st.warning(
            "Please enter a longer text to summarize."
        )
    else:
        try:
            device = next(model.parameters()).device

            inputs = tokenizer(
                article.strip(),
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            with st.spinner("Generating your summary..."):
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_words,
                        num_beams=4,
                        do_sample=False,
                        early_stopping=True
                    )

                summary = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True
                ).strip()

            st.subheader("Generated Summary")
            if summary:
                st.success("Summary generated successfully.")
                st.write(summary)

                st.download_button(
                    "Download Summary",
                    data=summary,
                    file_name="amharic_summary.txt",
                    mime="text/plain"
                )
            else:
                st.warning(
                    "The model returned an empty summary. "
                    "Try a different or shorter article."
                )

        except Exception as error:
            st.error("Unable to generate the summary.")
            st.exception(error)

st.divider()
st.caption(
    "AI-based Amharic summarization | "
    "Powered by a pretrained mT5 model"
)
