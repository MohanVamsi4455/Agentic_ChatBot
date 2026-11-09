import json
import uuid
import streamlit as st

# HumanMessage / AIMessage formats
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command  # used to resume a paused buy/sell

# Core chatbot & memory
from chatbot import chatbot, get_all_threads
# PDF upload -> per-chat FAISS index (the rag_search tool reads it)
from rag_tool import ingest_pdf, list_documents, delete_documents


# ---------------------------- page setup ----------------------------
st.set_page_config(
    page_title="🤖 Agentic ChatBot 🚀",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

USER_AVATAR = "🧑"
BOT_AVATAR = "🤖"

SUGGESTIONS = [
    ("💡", "Explain LangGraph in simple words"),
    ("🧠", "How does a checkpointer give a chatbot memory?"),
    ("📈", "What is the current stock price of AAPL?"),
    ("📄", "Summarize the attached PDF document"),
]

# Custom CSS for modern styling
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

    /* rounded buttons */
    .stButton > button {border-radius: 12px; transition: all .15s ease;}
    .stButton > button:hover {transform: translateY(-1px);}

    /* sidebar conversation buttons */
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


# ---------------------------- tool-call display ----------------------------
TOOL_LABELS = {
    "tavily_search":   ("🌐", "Searching the web",        "Searched the web",      "query"),
    "get_weather":     ("🌤️", "Checking the weather",     "Checked the weather",   "city"),
    "get_stock_price": ("📈", "Fetching stock price",     "Fetched stock price",   "symbol"),
    "perform_math":    ("🧮", "Calculating",              "Calculated",            None),
    "rag_search":      ("📄", "Searching your documents", "Searched your documents", "query"),
    "buy_stock":       ("🟢", "Placing buy order",        "Buy order handled",     "symbol"),
    "sell_stock":      ("🔴", "Placing sell order",       "Sell order handled",    "symbol"),
}


def tool_label(name, args, done):
    icon, running_text, done_text, detail_key = TOOL_LABELS.get(
        name, ("🔧", f"Using {name}", f"Used {name}", None)
    )
    text = done_text if done else running_text + "…"
    detail = (args or {}).get(detail_key) if detail_key else None
    if detail:
        text += f" · {detail}"
    return f"{icon} {text}"


def parse_args(args):
    """Tool args arrive as a dict (saved messages) or a JSON string (streamed chunks)."""
    if isinstance(args, dict):
        return args
    try:
        return json.loads(args) if args else {}
    except json.JSONDecodeError:
        return {}


def fill_tool_details(container, tool):
    """Show a tool call's input and result inside its expandable status box."""
    container.caption("Input")
    container.code(json.dumps(tool.get("args") or {}, indent=2), language="json")
    if tool.get("output") is not None:
        output = str(tool["output"])
        container.caption("Result")
        container.code(output[:1500] + ("…" if len(output) > 1500 else ""), language="json")


def render_finished_tool(tool):
    """A collapsed, already-finished tool call (used when redrawing old messages)."""
    box = st.status(
        tool_label(tool["name"], tool.get("args"), done=True),
        state="error" if tool.get("error") else "complete",
        expanded=False,
    )
    fill_tool_details(box, tool)


def load_conversation(thread_id):
    """Read a thread's saved messages from LangGraph's memory (SqliteSaver)."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("message", [])

    history = []
    tools = {}

    def flush(text=""):
        nonlocal tools
        if text or tools:
            history.append({"role": "assistant", "content": text, "tools": list(tools.values())})
        tools = {}

    for msg in messages:
        if isinstance(msg, HumanMessage):
            flush()
            history.append({"role": "user", "content": msg.text})
        elif isinstance(msg, ToolMessage):
            if msg.tool_call_id in tools:
                tools[msg.tool_call_id]["output"] = msg.content
                tools[msg.tool_call_id]["error"] = msg.status == "error"
        elif isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                tools[tc["id"]] = {"name": tc["name"], "args": tc["args"], "output": None}
            if msg.text and not msg.tool_calls:
                flush(msg.text)
    flush()
    return history


def get_pending_approval(config):
    """If a buy/sell tool paused this chat, return its payload (action, symbol, quantity)."""
    state = chatbot.get_state(config)
    for task in state.tasks:
        for intr in task.interrupts:
            return intr.value
    return None


def delete_thread(thread_id):
    """Remove a conversation from the sidebar and from LangGraph's memory."""
    st.session_state["chat_threads"].remove(thread_id)
    st.session_state["thread_titles"].pop(thread_id, None)
    if hasattr(chatbot.checkpointer, "delete_thread"):
        chatbot.checkpointer.delete_thread(thread_id)
    delete_documents(thread_id)
    if thread_id == st.session_state["thread_id"]:
        if st.session_state["chat_threads"]:
            switch_thread(st.session_state["chat_threads"][-1])
        else:
            reset_chat()


def switch_thread(thread_id):
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = load_conversation(thread_id)


def make_title(text):
    text = text.strip().replace("\n", " ")
    return text[:28] + ("…" if len(text) > 28 else "")


def thread_title(thread_id):
    """Title = first user message. Falls back to saved DB state after a restart."""
    titles = st.session_state["thread_titles"]
    if thread_id not in titles:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        first_user = next(
            (m for m in state.values.get("message", []) if isinstance(m, HumanMessage)),
            None,
        )
        if first_user is None:
            return "New chat"
        titles[thread_id] = make_title(first_user.text)
    return titles[thread_id]


# ---------------------------- session state ----------------------------
defaults = {
    "message_history": [],
    "chat_threads": get_all_threads(),
    "thread_titles": {},
    "pending_prompt": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_uuid()

add_thread(st.session_state["thread_id"])

config = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {"thread_id": st.session_state["thread_id"]},
    "run_name": "chat_trace",
}


# ---------------------------- sidebar ----------------------------
with st.sidebar:
    st.markdown("## 🤖 Agentic ChatBot 🚀")
    st.caption("⚡ Powered by LangGraph + Gemini")

    if st.button("➕  New chat", use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()

    st.markdown('<div class="sidebar-label">Conversations</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="sidebar-label">Documents in this chat</div>', unsafe_allow_html=True)
    current_docs = list_documents(st.session_state["thread_id"])
    if current_docs:
        for doc in current_docs:
            st.caption(f"📄 **{doc['name']}** · {doc['pages']} pages")
    else:
        st.caption("None yet. Attach a PDF with 📎 in the message box.")

    st.divider()
    st.caption(f"{len(st.session_state['chat_threads'])} conversation(s) · persistent memory enabled")


# ---------------------------- main chat area ----------------------------
st.markdown('<p class="app-title">🤖 Agentic ChatBot 🚀 ✨</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">⚡ Powered by LangGraph + Gemini · Persistent Memory 🧠</p>',
    unsafe_allow_html=True,
)

# Empty chat -> show welcome card
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

# Redraw messages
for message in st.session_state["message_history"]:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("files"):
            st.caption("📎 " + " · ".join(message["files"]))
        for tool in message.get("tools", []):
            render_finished_tool(tool)
        if message["content"]:
            st.markdown(message["content"])


# ---------------------------- Human-In-The-Loop (HITL) ----------------------------
pending = get_pending_approval(config)
resume_value = None
if pending:
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.warning(
            f"🛡️ **Approval needed:** {pending['action']} {pending['quantity']} "
            f"shares of **{pending['symbol']}**. Do you want to place this order?"
        )
        col_yes, col_no = st.columns(2)
        if col_yes.button("✅ Approve", key="hitl-approve", type="primary", use_container_width=True):
            resume_value = "yes"
        if col_no.button("❌ Reject", key="hitl-reject", use_container_width=True):
            resume_value = "no"

# Input box (disabled during pending approval to prevent conflict)
submitted = st.chat_input(
    "Message Agentic ChatBot… (📎 attach a PDF to chat with it)",
    accept_file="multiple",
    file_type=["pdf"],
    disabled=bool(pending),
)

if submitted:
    user_message = (submitted.text or "").strip()
    uploaded_files = list(submitted.files or [])
else:
    user_message = st.session_state["pending_prompt"] or ""
    uploaded_files = []
st.session_state["pending_prompt"] = None

if user_message or uploaded_files:
    thread_id = st.session_state["thread_id"]
    if thread_id not in st.session_state["thread_titles"]:
        st.session_state["thread_titles"][thread_id] = make_title(
            user_message or uploaded_files[0].name
        )

    file_names = [f.name for f in uploaded_files]
    st.session_state["message_history"].append(
        {"role": "user", "content": user_message, "files": file_names}
    )
    with st.chat_message("user", avatar=USER_AVATAR):
        if file_names:
            st.caption("📎 " + " · ".join(file_names))
        if user_message:
            st.markdown(user_message)

    # Ingest uploaded PDFs
    if uploaded_files:
        notes = []
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            for f in uploaded_files:
                with st.status(f"📄 Reading {f.name}…", expanded=False) as box:
                    try:
                        info = ingest_pdf(f.getvalue(), f.name, thread_id)
                        box.update(
                            label=f"✅ {f.name} is ready · {info['pages']} pages",
                            state="complete",
                        )
                        notes.append(f"📄 Added **{f.name}** ({info['pages']} pages) to this chat.")
                    except Exception as e:
                        box.update(label=f"❌ Couldn't read {f.name}", state="error")
                        box.write(str(e))
                        notes.append(f"❌ Couldn't read **{f.name}**: {e}")
        if not user_message:
            notes.append("Ask me anything about it.")
        st.session_state["message_history"].append(
            {"role": "assistant", "content": "\n\n".join(notes)}
        )

# Execute or resume LLM turn
if user_message or resume_value:
    if resume_value:
        graph_input = Command(resume=resume_value)
    else:
        graph_input = {"message": [HumanMessage(content=user_message)]}

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        tools_area = st.container()
        thinking = st.empty()
        thinking.caption("⏳ Thinking…")

        tools = {}
        boxes = {}
        chunk_ids = {}

        def ai_tokens():
            for message_chunk, metadata in chatbot.stream(
                graph_input,
                config=config,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, AIMessageChunk) and message_chunk.tool_call_chunks:
                    for tc in message_chunk.tool_call_chunks:
                        key = (message_chunk.id, tc.get("index"))
                        call_id = tc.get("id") or chunk_ids.get(key)
                        if call_id is None:
                            continue
                        chunk_ids[key] = call_id
                        if call_id not in tools:
                            tools[call_id] = {
                                "name": tc.get("name") or "tool",
                                "args": "",
                                "output": None,
                            }
                        tools[call_id]["args"] += tc.get("args") or ""
                        args = parse_args(tools[call_id]["args"])
                        label = tool_label(tools[call_id]["name"], args, done=False)
                        if call_id not in boxes:
                            thinking.empty()
                            boxes[call_id] = tools_area.status(label, state="running", expanded=False)
                        else:
                            boxes[call_id].update(label=label)

                elif isinstance(message_chunk, ToolMessage):
                    call_id = message_chunk.tool_call_id
                    tool = tools.get(call_id)
                    if tool is None:
                        continue
                    tool["args"] = parse_args(tool["args"])
                    tool["output"] = message_chunk.content
                    tool["error"] = message_chunk.status == "error"
                    box = boxes[call_id]
                    fill_tool_details(box, tool)
                    box.update(
                        label=tool_label(tool["name"], tool["args"], done=True),
                        state="error" if tool["error"] else "complete",
                    )
                    thinking.caption("⏳ Thinking…")

                elif isinstance(message_chunk, (AIMessageChunk, AIMessage)) and message_chunk.text:
                    thinking.empty()
                    yield message_chunk.text

        try:
            tokens = ai_tokens()
            first_token = next(tokens, None)

            def all_tokens():
                if first_token:
                    yield first_token
                yield from tokens

            ai_message = st.write_stream(all_tokens()) if first_token else ""
        except Exception as e:
            ai_message = None
            st.error(f"Sorry, something went wrong: {e}")
        finally:
            thinking.empty()

    finished = [t for t in tools.values() if t["output"] is not None]
    if ai_message or finished:
        for tool in finished:
            tool["args"] = parse_args(tool["args"])
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message or "", "tools": finished}
        )

    st.rerun()

elif uploaded_files:
    st.rerun()

