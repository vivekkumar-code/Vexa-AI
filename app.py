import streamlit as st
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Vexa AI",
    page_icon="🤖",
    layout="centered"
)


# ---------------- DATABASE ----------------

DB_NAME = "vexa_history.db"


def get_db():
    return sqlite3.connect(DB_NAME)


def create_database():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def create_chat(title="New Chat"):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO chats (title, created_at) VALUES (?, ?)",
        (title, datetime.now().isoformat())
    )

    chat_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return chat_id


def save_message(chat_id, role, content):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO messages
        (chat_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            chat_id,
            role,
            content,
            datetime.now().isoformat()
        )
    )

    conn.commit()
    conn.close()


def get_chats():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, title FROM chats ORDER BY id DESC"
    )

    chats = cursor.fetchall()

    conn.close()

    return chats


def get_messages(chat_id):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, content
        FROM messages
        WHERE chat_id = ?
        ORDER BY id
        """,
        (chat_id,)
    )

    messages = cursor.fetchall()

    conn.close()

    return messages


# ---------------- DATABASE START ----------------

create_database()


# ---------------- SESSION STATE ----------------

if "chat_id" not in st.session_state:

    st.session_state.chat_id = create_chat()


if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------- MODEL ----------------

@st.cache_resource
def load_model():

    HF_TOKEN = st.secrets["HF_TOKEN"]

    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        temperature=1.0,
        huggingfacehub_api_token=HF_TOKEN
    )

    return ChatHuggingFace(llm=llm)


model = load_model()


# ---------------- SYSTEM MESSAGE ----------------

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
Always reply in the same language as the user.
If the user asks in Hindi, reply in Hindi.
If the user asks in English, reply in English.
If the user asks in Hinglish, reply in Hinglish.
Never use Chinese or any other language unless the user asks for it.

IMPORTANT:
Never guess the current date, current day, or current time.
The application handles date, day and time separately.
"""
)


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("🤖 Vexa AI")

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.chat_id = create_chat()
        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.subheader("🕘 History")

    chats = get_chats()

    for chat_id, title in chats:

        if st.button(
            title,
            key=f"chat_{chat_id}",
            use_container_width=True
        ):

            st.session_state.chat_id = chat_id

            db_messages = get_messages(chat_id)

            st.session_state.messages = [
                {
                    "role": role,
                    "content": content
                }
                for role, content in db_messages
            ]

            st.rerun()


# ---------------- MAIN UI ----------------

st.title("🤖 Vexa AI")

st.write("Ask me anything!")


# ---------------- DISPLAY HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ---------------- CHAT INPUT ----------------

question = st.chat_input(
    "Ask Vexa AI anything..."
)


if question:

    # ---------------- SAVE USER MESSAGE ----------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    save_message(
        st.session_state.chat_id,
        "user",
        question
    )


    # ---------------- SHOW USER MESSAGE ----------------

    with st.chat_message("user"):

        st.write(question)


    # =====================================================
    # CURRENT INDIA DATE / DAY / TIME
    # =====================================================

    india_now = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    current_date = india_now.strftime(
        "%d %B %Y"
    )

    current_day = india_now.strftime(
        "%A"
    )

    current_time = india_now.strftime(
        "%I:%M %p"
    )


    # Convert question to lowercase

    q = question.lower().strip()


    # =====================================================
    # DATE DETECTION
    # =====================================================

    date_words = [
        "date",
        "tareekh",
        "तारीख"
    ]

    date_context = [
        "aaj",
        "today",
        "current",
        "abhi",
        "batao",
        "bata",
        "kya hai"
    ]

    is_date_question = (
        any(word in q for word in date_words)
        and any(word in q for word in date_context)
    )


    # =====================================================
    # DAY DETECTION
    # =====================================================

    day_words = [
        "day",
        "din",
        "वार",
        "दिन"
    ]

    day_context = [
        "aaj",
        "today",
        "current",
        "kaun",
        "konsa",
        "kaunsa",
        "kya"
    ]

    is_day_question = (
        any(word in q for word in day_words)
        and any(word in q for word in day_context)
    )


    # =====================================================
    # TIME DETECTION
    # =====================================================

    time_words = [
        "time",
        "samay",
        "baje",
        "baj",
        "समय"
    ]

    time_context = [
        "abhi",
        "current",
        "now",
        "kya",
        "kitne",
        "batao",
        "bata"
    ]

    is_time_question = (
        any(word in q for word in time_words)
        and any(word in q for word in time_context)
    )


    # =====================================================
    # DATE RESPONSE
    # =====================================================

    if is_date_question:

        answer = (
            f"Aaj ki date {current_date} hai."
        )


    # =====================================================
    # DAY RESPONSE
    # =====================================================

    elif is_day_question:

        answer = (
            f"Aaj {current_day} hai."
        )


    # =====================================================
    # TIME RESPONSE
    # =====================================================

    elif is_time_question:

        answer = (
            f"Abhi India mein "
            f"{current_time} IST ho raha hai."
        )


    # =====================================================
    # NORMAL AI QUESTION
    # =====================================================

    else:

        messages = [system_message]

        for message in st.session_state.messages:

            if message["role"] == "user":

                messages.append(
                    HumanMessage(
                        content=message["content"]
                    )
                )

            elif message["role"] == "assistant":

                messages.append(
                    AIMessage(
                        content=message["content"]
                    )
                )


        # ---------------- GENERATE RESPONSE ----------------

        with st.spinner("Thinking..."):

            try:

                result = model.invoke(messages)

                answer = result.content

            except Exception as e:

                answer = f"Error: {e}"


    # =====================================================
    # SHOW ASSISTANT RESPONSE
    # =====================================================

    with st.chat_message("assistant"):

        st.write(answer)


    # ---------------- SAVE ASSISTANT RESPONSE ----------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    save_message(
        st.session_state.chat_id,
        "assistant",
        answer
    )


    