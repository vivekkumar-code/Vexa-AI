from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import load_prompt
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os

load_dotenv()

# ---------------- LLM ----------------
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=1000,
    temperature=1.0
)

chat_model = ChatHuggingFace(llm=llm)

research_template = load_prompt("template.json")

# ---------------- History File ----------------
HISTORY_FILE = "history.json"


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except:
            return []
    return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=4)


# ---------------- Session State ----------------
if "history" not in st.session_state:
    st.session_state.history = load_history()

if "current_chat" not in st.session_state:
    st.session_state.current_chat = []


# ---------------- New Chat Function ----------------
def new_chat():
    st.session_state.current_chat = []


# ---------------- UI ----------------
st.header("🔬 Research Tool")

# Sidebar
with st.sidebar:

    st.subheader("💬 Chat")

    if st.button("➕ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.markdown("---")

    st.subheader("🕘 History")

    if st.session_state.history:

        for i, chat in enumerate(
            reversed(st.session_state.history)
        ):

            title = chat.get("question", "General Summary")

            if len(title) > 35:
                title = title[:35] + "..."

            timestamp = chat.get(
                "timestamp",
                "Unknown time"
            )

            if st.button(
                f"🗨️ {title}\n{timestamp}",
                key=f"history_{i}",
                use_container_width=True
            ):

                st.session_state.current_chat = [chat]
                st.rerun()

    else:
        st.caption("No previous chats yet.")


# ---------------- Research Inputs ----------------
paper_input = st.selectbox(
    "Select Research Paper Name (optional)",
    [
        "None (Ask a general question instead)",

        # NLP / Transformers / LLM
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "XLNet: Generalized Autoregressive Pretraining for Language Understanding",
        "ALBERT: A Lite BERT for Self-supervised Learning",
        "Language Models are Few-Shot Learners",
        "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        "LLaMA: Open and Efficient Foundation Language Models",
        "LLaMA 2: Open Foundation and Fine-Tuned Chat Models",
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",

        # Computer Vision
        "ImageNet Classification with Deep Convolutional Neural Networks",
        "Very Deep Convolutional Networks for Large-Scale Image Recognition",
        "Going Deeper with Convolutions",
        "Deep Residual Learning for Image Recognition",
        "You Only Look Once: Unified, Real-Time Object Detection",
        "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
        "Mask R-CNN",
        "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "Training data-efficient image transformers & distillation through attention",
        "Segment Anything",

        # Generative AI
        "Generative Adversarial Nets",
        "Auto-Encoding Variational Bayes",
        "Denoising Diffusion Probabilistic Models",
        "High-Resolution Image Synthesis with Latent Diffusion Models",
        "Neural Discrete Representation Learning",

        # Multimodal AI
        "Learning Transferable Visual Models From Natural Language Supervision",
        "BLIP: Bootstrapping Language-Image Pre-training",
        "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models",
        "Visual Instruction Tuning",

        # Reinforcement Learning
        "Playing Atari with Deep Reinforcement Learning",
        "Human-level Control Through Deep Reinforcement Learning",
        "Mastering the Game of Go with Deep Neural Networks",
        "Mastering the Game of Go without Human Knowledge",
        "Proximal Policy Optimization Algorithms",

        # Classical / Machine Learning
        "Random Forests",
        "XGBoost: A Scalable Tree Boosting System",
        "Support-Vector Networks",

        # Other Important Deep Learning
        "Neural Machine Translation by Jointly Learning to Align and Translate",
        "Sequence to Sequence Learning with Neural Networks",
        "NeRF: Representing Scenes as Neural Radiance Fields"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)

language_input = st.selectbox(
    "Select Output Language",
    [
        "English",
        "Hindi",
        "Hinglish",
        "Spanish",
        "French"
    ]
)

st.markdown("---")


# ---------------- Show Current Chat ----------------
for qa in st.session_state.current_chat:

    st.markdown(
        f"**🕒 {qa.get('timestamp', '')}**"
    )

    st.markdown(
        f"**Q: {qa['question']}**"
    )

    st.write(qa["answer"])

    st.markdown("---")


# ---------------- Question Form ----------------
with st.form(
    key=f"qa_form_{len(st.session_state.current_chat)}"
):

    custom_question = st.text_input(
        "Ask your question"
    )

    submitted = st.form_submit_button(
        "Summarize"
    )


# ---------------- Generate Answer ----------------
if submitted:

    if paper_input == "None (Ask a general question instead)":

        paper_section = (
            "This is a general question, "
            "not tied to any specific research paper."
        )

    else:

        paper_section = (
            f'Explain this in context of the research '
            f'paper titled "{paper_input}".'
        )


    if custom_question.strip():

        question_section = (
            f"Answer this question: {custom_question}"
        )

        question_to_save = custom_question

    else:

        question_section = (
            "Provide a general summary/explanation."
        )

        question_to_save = "(General summary)"


    # ---------------- Prompt ----------------
    prompt = research_template.invoke({

        "paper_section": paper_section,

        "question_section": question_section,

        "style_input": style_input,

        "length_input": length_input,

        "language_input": language_input
    })


    # ---------------- LLM Response ----------------
    result = chat_model.invoke(prompt)


    # ---------------- Indian Date & Time ----------------
    india_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    timestamp = india_time.strftime(
        "%d %b %Y, %I:%M:%S %p"
    )


    # ---------------- Chat Data ----------------
    chat_data = {

        "question": question_to_save,

        "answer": result.content,

        "timestamp": timestamp,

        "paper": paper_input,

        "style": style_input,

        "length": length_input,

        "language": language_input
    }


    # Current chat me add
    st.session_state.current_chat.append(
        chat_data
    )


    # Permanent history me add
    st.session_state.history.append(
        chat_data
    )


    # history.json me save
    save_history(
        st.session_state.history
    )


    st.rerun()