import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    st.error("GROQ_API_KEY is missing from the .env file.")
    st.stop()

client = Groq(api_key=API_KEY)

MODEL = "llama-3.3-70b-versatile"


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.set_page_config(
    page_title="F1 Strategy Copilot",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ F1 Strategy Copilot")
st.caption("Multi-Agent Generative AI System for Formula 1 Strategy Analysis")


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

question = st.text_area(
    "Ask your F1 strategy question:",
    placeholder="Example: Why is an undercut useful in Formula 1?",
    height=120
)


# --------------------------------------------------
# F1 PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are an F1 race strategy assistant.

Your job is to explain Formula 1 race strategy
clearly and accurately.

Rules:

1. Explain technical terms when necessary.
2. Separate known facts from interpretation.
3. Do not invent race data.
4. If specific race data is unavailable, clearly say so.
5. Do not pretend to know a team's internal strategy
   unless reliable information is provided.
6. Give concise but useful explanations.
"""


# --------------------------------------------------
# GENERATE RESPONSE
# --------------------------------------------------

if st.button("🔍 Analyze Strategy", type="primary"):

    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    with st.spinner("Analyzing your question..."):

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content

    st.subheader("🧠 F1 Strategy Analysis")

    st.write(answer)