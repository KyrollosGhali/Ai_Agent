import re

import streamlit as st
from langchain_core.messages import HumanMessage

from agent import app


DEFAULT_PROMPT = (
    "A cinematic, photorealistic medium shot of a young woman with pink-gold hair "
    "in a cozy late-90s bedroom at golden hour, soft lens flare, shallow depth of field"
)


def extract_urls(text: str) -> list[str]:
    pattern = r'https?://[^\s)\]>\"]+'
    return re.findall(pattern, text or "")


st.set_page_config(page_title="Image Agent", page_icon="🖼️", layout="centered")
st.title("Image Generation Agent")
st.caption("Type a prompt and run your LangGraph + Groq + Replicate pipeline.")

prompt = st.text_area(
    "Prompt",
    value=DEFAULT_PROMPT,
    height=180,
    placeholder="Describe the image you want...",
)

run_clicked = st.button("Generate", type="primary")

if run_clicked:
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        st.warning("Please enter a prompt before generating.")
    else:
        with st.spinner("Generating image..."):
            try:
                result = app.invoke({"messages": [HumanMessage(content=cleaned_prompt)]})
                final_text = str(result["messages"][-1])
            except Exception as exc:
                st.error(f"Workflow failed: {exc}")
            else:
                st.subheader("Agent Response")
                st.write(final_text)

                urls = extract_urls(final_text)
                if urls:
                    st.success("Image generated successfully!")
                    st.subheader("Detected Image URLs")
                    for url in urls:
                        st.markdown(f"- [{url}]({url})")
                        st.image(url, caption="Generated image", use_container_width=True)