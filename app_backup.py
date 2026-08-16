import streamlit as st
import sqlite3

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()


# =========================
# Page Settings
# =========================

st.set_page_config(
    page_title="Vexa AI",
    page_icon="🤖"
)

st.title("🤖 Vexa AI")
st.write("Ask me anything!")


# =========================
# Database
# =========================

conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL
)
""")

conn.commit()


def save_message(role, content):
    cursor.execute(
        "INSERT INTO chat_history (role, content) VALUES (?, ?)",
        (role, content)
    )
    conn.commit()


def get_history():
    cursor.execute(
        "SELECT role, content FROM chat_history ORDER BY id"
    )
    return cursor.fetchall()


# =========================
# AI Model
# =========================

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)


# =========================
# System Message
# =========================

system_message = SystemMessage(
    content="""
You are Vexa AI, a helpful and friendly AI assistant.

Your name is Vexa AI.
Your father's name is Vivek Engineer.

If the user asks your name, say:
"My name is Vexa AI."

If the user asks who your father is, say:
"My father's name is Vivek Engineer."

Answer all other questions normally, clearly and helpfully.
"""
)


# =========================
# Show Previous Chat History
# =========================

history = get_history()

for role, content in history:

    if role == "user":
        with st.chat_message("user"):
            st.write(content)

    elif role == "assistant":
        with st.chat_message("assistant"):
            st.write(content)


# =========================
# New Question
# =========================

question = st.chat_input("Ask Vexa AI anything...")


if question:

    # Save user message
    save_message("user", question)

    with st.chat_message("user"):
        st.write(question)

    # Get previous messages for AI context
    messages = [system_message]

    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content=content))

    messages.append(HumanMessage(content=question))

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = model.invoke(messages)

            answer = result.content

            st.write(answer)

            # Save AI response
            save_message("assistant", answer)          