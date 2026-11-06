# Import the LangGraph chatbot we built in agentic_chatbot.py
from agentic_chatbot import chatbot
# HumanMessage = a message written by the user (LangChain's message format)
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
# Streamlit = the library that builds the web page (UI)
import streamlit as st
import uuid

# NOTE: Streamlit re-runs this whole file from top to bottom every time
# the user does something. Normal variables are reset on every run,
# so anything we want to keep must be stored in st.session_state.


# ---------------------------- page setup ----------------------------

# Must be the first Streamlit call: browser-tab title, icon and layout
st.set_page_config(
    page_title="Agentic ChatBot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

USER_AVATAR = "🧑"
BOT_AVATAR = "🤖"

SUGGESTIONS = [
    ("💡", "Explain LangGraph in simple words"),
    ("🧠", "How does a checkpointer give a chatbot memory?"),
    ("🐍", "Write a Python function to reverse a string"),
    ("✍️", "Give me 3 ideas for an AI side project"),
]

# Custom CSS to make the page look cleaner
st.markdown(
    """
    <style>
    /* hide Streamlit's default menu & footer */
    #MainMenu, footer {visibility: hidden;}

    .block-container {padding-top: 2rem; padding-bottom: 6rem; max-width: 820px;}

    /* gradient title */
    .app-title {
        font-size: 2.1rem; font-weight: 800; margin-bottom: 0;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .app-subtitle {color: #8b8fa3; margin-top: .2rem; margin-bottom: 1.5rem;}

    /* welcome card shown on an empty chat */
    .welcome {
        text-align: center; padding: 2.2rem 1rem 1.2rem;
        border: 1px solid rgba(128,128,128,.2); border-radius: 18px;
        background: linear-gradient(180deg, rgba(99,102,241,.08), rgba(236,72,153,.04));
        margin-bottom: 1.2rem;
    }
    .welcome h3 {margin: .4rem 0 .3rem;}
    .welcome p {color: #8b8fa3; margin: 0;}

    /* chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 16px; padding: .8rem 1rem; margin-bottom: .6rem;
        border: 1px solid rgba(128,128,128,.15);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
        background: rgba(99,102,241,.08);
    }

    /* rounded buttons everywhere */
    .stButton > button {border-radius: 12px; transition: all .15s ease;}
    .stButton > button:hover {transform: translateY(-1px);}

    /* sidebar conversation buttons: left-aligned text */
    section[data-testid="stSidebar"] .stButton > button {justify-content: flex-start; text-align: left;}
    section[data-testid="stSidebar"] .stButton > button p {
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .sidebar-label {
        font-size: .75rem; font-weight: 700; letter-spacing: .06em;
        text-transform: uppercase; color: #8b8fa3; margin: 1rem 0 .4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------- helper functions ----------------------------

def generate_uuid():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    """Start a brand-new conversation with a fresh thread_id."""
    st.session_state["thread_id"] = generate_uuid()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    """Read a thread's saved messages from LangGraph's memory (MemorySaver)."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("message", [])

    history = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        history.append({"role": role, "content": msg.text})
    return history


def delete_thread(thread_id):
    """Remove a conversation from the sidebar and from LangGraph's memory."""
    st.session_state["chat_threads"].remove(thread_id)
    st.session_state["thread_titles"].pop(thread_id, None)
    if hasattr(chatbot.checkpointer, "delete_thread"):
        chatbot.checkpointer.delete_thread(thread_id)
    # If we deleted the chat we're looking at, open the newest one (or a new one)
    if thread_id == st.session_state["thread_id"]:
        if st.session_state["chat_threads"]:
            switch_thread(st.session_state["chat_threads"][-1])
        else:
            reset_chat()


def switch_thread(thread_id):
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = load_conversation(thread_id)


def thread_title(thread_id):
    return st.session_state["thread_titles"].get(thread_id, "New chat")


# ---------------------------- session state ----------------------------

# On the very first run, create the things we need to remember
defaults = {
    "message_history": [],
    "chat_threads": [],
    "thread_titles": {},      # thread_id -> short title (first user message)
    "pending_prompt": None,   # set when a suggestion card is clicked
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_uuid()

add_thread(st.session_state["thread_id"])

# thread_id tells LangGraph's memory (MemorySaver) which conversation this is,
# so the bot remembers earlier messages in the same thread
config = {"configurable": {"thread_id": st.session_state["thread_id"]}}


# ---------------------------- sidebar ----------------------------

with st.sidebar:
    st.markdown("## 🤖 Agentic ChatBot")
    st.caption("Powered by LangGraph + Gemini")

    if st.button("➕  New chat", use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()

    st.markdown('<div class="sidebar-label">Conversations</div>', unsafe_allow_html=True)

    # Newest conversation on top; click to open, 🗑 to delete
    for thread_id in reversed(st.session_state["chat_threads"]):
        is_current = thread_id == st.session_state["thread_id"]
        col_open, col_delete = st.columns([5, 1], vertical_alignment="center")

        label = ("💬 " if is_current else "🗨️ ") + thread_title(thread_id)
        if col_open.button(
            label,
            key=f"open-{thread_id}",
            type="secondary",
            disabled=is_current,
            use_container_width=True,
        ):
            switch_thread(thread_id)
            st.rerun()

        if col_delete.button("🗑", key=f"del-{thread_id}", help="Delete this chat"):
            delete_thread(thread_id)
            st.rerun()

    st.divider()
    st.caption(f"{len(st.session_state['chat_threads'])} conversation(s) · memory lasts while the app runs")


# ---------------------------- main chat area ----------------------------

st.markdown('<p class="app-title">Agentic ChatBot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Ask anything — each conversation keeps its own memory.</p>',
    unsafe_allow_html=True,
)

# Empty chat -> show a welcome card with clickable suggestions
if not st.session_state["message_history"]:
    st.markdown(
        """
        <div class="welcome">
            <div style="font-size:2.6rem">👋</div>
            <h3>How can I help you today?</h3>
            <p>Pick a suggestion below or type your own question.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (icon, text) in enumerate(SUGGESTIONS):
        if cols[i % 2].button(f"{icon}  {text}", key=f"sugg-{i}", use_container_width=True):
            st.session_state["pending_prompt"] = text
            st.rerun()

# Draw every old message again (because the page is rebuilt on each run)
for message in st.session_state["message_history"]:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# The text box at the bottom of the page.
# It returns the typed text after Enter is pressed, otherwise None.
typed_message = st.chat_input("Message Agentic ChatBot…")

# A clicked suggestion counts as a message too
user_message = typed_message or st.session_state["pending_prompt"]
st.session_state["pending_prompt"] = None

# This block runs only when the user has sent a message
if user_message:
    # Use the first message as this conversation's sidebar title
    thread_id = st.session_state["thread_id"]
    if thread_id not in st.session_state["thread_titles"]:
        title = user_message.strip().replace("\n", " ")
        st.session_state["thread_titles"][thread_id] = title[:28] + ("…" if len(title) > 28 else "")

    # Save + show the user's message
    st.session_state["message_history"].append({"role": "user", "content": user_message})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_message)

    # Only pass on the LLM's text chunks (skip anything that isn't an AI chunk)
    def ai_tokens():
        for message_chunk, metadata in chatbot.stream(
            {"message": [HumanMessage(content=user_message)]},
            config=config,
            stream_mode="messages",  # yields (token_chunk, metadata) as the LLM generates
        ):
            if isinstance(message_chunk, (AIMessageChunk, AIMessage)) and message_chunk.text:
                yield message_chunk.text

    # Stream the bot's reply token by token into an assistant chat bubble
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            with st.spinner("Thinking…"):
                ai_message = st.write_stream(ai_tokens())
        except Exception as e:
            ai_message = None
            st.error(f"Sorry, something went wrong: {e}")

    # Save the bot's reply to the history
    if ai_message:
        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    # Rerun so the sidebar shows the new title
    st.rerun()
