import os
import sqlite3
import logging
import warnings
import requests
from datetime import datetime, timezone
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, message_chunk_to_message
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

# PDF RAG tools and helpers
from rag_tool import rag_search, list_documents, get_env_var

logging.getLogger("google_genai").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

load_dotenv()


# Ensure TAVILY_API_KEY is available in os.environ if set via Streamlit secrets
tavily_key = get_env_var("TAVILY_API_KEY")
if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key


@tool
def tavily_search(query: str) -> dict:
    """Search the web for latest news and information."""
    return TavilySearch(max_results=2).invoke({"query": query})


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    api_key = get_env_var("WEATHER_STACK_API_KEY") or get_env_var("OPENWEATHER_API_KEY")
    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
    try:
        res = requests.get(url, timeout=10)
        return res.json()
    except Exception as e:
        return {"error": f"Failed to retrieve weather for {city}: {str(e)}"}


def _yahoo_quote(symbol: str) -> dict | None:
    """Fallback quote from Yahoo Finance. Yahoo uses .BO for BSE (not .BSE)."""
    yahoo_symbol = symbol.replace(".BSE", ".BO")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval=1d&range=5d"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        result = (r.json().get("chart") or {}).get("result")
    except (requests.RequestException, ValueError):
        return None
    if not result:
        return None
    meta = result[0]["meta"]
    return {
        "symbol": symbol,
        "price": meta["regularMarketPrice"],
        "currency": meta.get("currency"),
        "as_of": datetime.fromtimestamp(meta["regularMarketTime"], tz=timezone.utc).date().isoformat(),
        "source": "Yahoo Finance",
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """Get the latest stock price.
    US stocks: plain ticker (AAPL, TSLA).
    Indian stocks: use the BSE suffix, e.g. SBIN.BSE, RELIANCE.BSE, TCS.BSE (NOT .NS).
    Prices for .BSE symbols are in INR. Always tell the user the 'as_of' date."""
    av_key = get_env_var("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={av_key}"
    try:
        data = requests.get(url, timeout=10).json()
        quote = data.get("Global Quote")
        if quote:
            return {
                "symbol": quote["01. symbol"],
                "price": float(quote["05. price"]),
                "change_percent": quote["10. change percent"],
                "as_of": quote["07. latest trading day"],
                "source": "Alpha Vantage",
            }
    except Exception:
        pass

    # Alpha Vantage does not cover every Indian stock (e.g. JIOFIN). Try Yahoo next.
    fallback = _yahoo_quote(symbol)
    if fallback:
        return fallback
    return {
        "error": f"No price found for '{symbol}' in Alpha Vantage or Yahoo Finance. "
                 "Indian stocks need the .BSE suffix (e.g. SBIN.BSE)."
    }


@tool
def perform_math(a: float, op: str, b: float = 0) -> float:
    """Do math. op is one of: add, sub, mul, div, pow, mod, max, min, abs (abs uses only a)."""
    if op == "add": return a + b
    if op == "sub": return a - b
    if op == "mul": return a * b
    if op == "div": return a / b if b != 0 else float("inf")
    if op == "pow": return a ** b
    if op == "mod": return a % b if b != 0 else 0
    if op == "max": return max(a, b)
    if op == "min": return min(a, b)
    if op == "abs": return abs(a)
    return 0.0


# ---------------------------- Human-In-The-Loop (HITL) ----------------------------
MAX_ORDER_SHARES = 1000  # hard limit enforced in code, not in the prompt


def _trade(action: str, symbol: str, quantity: int) -> dict:
    """Shared logic for buy and sell: check the order, then pause for a human."""
    symbol = symbol.upper()
    if quantity <= 0 or quantity > MAX_ORDER_SHARES:
        return {
            "status": "rejected",
            "message": f"Quantity must be between 1 and {MAX_ORDER_SHARES} shares.",
        }

    # Pauses the graph. The UI shows this payload and resumes with "yes" or "no".
    # On resume this line returns the answer instead of pausing.
    decision = interrupt({"action": action, "symbol": symbol, "quantity": quantity})

    if decision == "yes":
        # Simulated execution for demo / portfolio tracking
        return {
            "status": "filled (simulated)",
            "action": action,
            "symbol": symbol,
            "quantity": quantity,
        }
    return {
        "status": "cancelled",
        "message": f"{action} of {quantity} {symbol} was declined by the user.",
    }


@tool
def buy_stock(symbol: str, quantity: int) -> dict:
    """Buy shares of a stock. The user must approve the order before it is placed."""
    return _trade("BUY", symbol, quantity)


@tool
def sell_stock(symbol: str, quantity: int) -> dict:
    """Sell shares of a stock. The user must approve the order before it is placed."""
    return _trade("SELL", symbol, quantity)


# Full tool suite
tools = [
    tavily_search,
    get_stock_price,
    perform_math,
    get_weather,
    rag_search,
    buy_stock,
    sell_stock,
]


def make_llm(model: str):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.0,
        api_key=get_env_var("GEMINI_API_KEY") or get_env_var("GOOGLE_API_KEY"),
        thinking_level="minimal",  # skip long internal thinking for snappy first token
        timeout=60,
        max_retries=1,
    )


# Bind tools to EACH model, then configure fallback
llm = make_llm("gemini-3.1-flash-lite").bind_tools(tools).with_fallbacks(
    [make_llm("gemini-3-flash-preview").bind_tools(tools)]
)


class Message(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]


def system_prompt(thread_id: str | None) -> SystemMessage:
    """Tell the model which tools to prefer, and which PDFs this chat has."""
    text = (
        "You are a helpful assistant. Use the tools when they help: web search for "
        "news, weather, stock prices and math. To buy or sell shares, call buy_stock "
        "or sell_stock. The user is then asked to approve, so do not claim the order "
        "is placed until the tool returns a filled status."
    )
    docs = list_documents(thread_id) if thread_id else []
    if docs:
        names = ", ".join(d["name"] for d in docs)
        text += (
            f"\nThe user has uploaded these PDF documents in this chat: {names}. "
            "For ANY question about them (or about 'the document', 'the file', 'the PDF'), "
            "call rag_search first and answer only from the passages it returns. "
            "Cite the file name and page number, e.g. (report.pdf, p. 3). "
            "If the passages don't contain the answer, say so."
        )
    return SystemMessage(content=text)


def llm_call(state: Message, config: RunnableConfig):
    thread_id = config.get("configurable", {}).get("thread_id")
    message = [system_prompt(thread_id)] + state["message"]

    cancel_event = config.get("configurable", {}).get("cancel_event")

    response = None
    cancelled = False
    stream = llm.stream(message, config)
    try:
        for chunk in stream:
            response = chunk if response is None else response + chunk
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
    finally:
        stream.close()

    if response is None:
        return {"message": []}
    if cancelled:
        return {"message": [AIMessage(content=response.content, id=response.id)]}
    return {"message": [message_chunk_to_message(response)]}


tool_node = ToolNode(tools, messages_key="message")


def route_tools(state: Message):
    """Go to 'tools' if the last AI message asked for a tool, otherwise END."""
    return tools_condition(state, messages_key="message")


# ---------------------------- Memory & Graph Compilation ----------------------------
DATA_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "chatbot.db")
connection = sqlite3.connect(database=DB_PATH, check_same_thread=False)

checkpoint = SqliteSaver(connection)
graph = StateGraph(Message)
graph.add_node("BOT", llm_call)
graph.add_node("tools", tool_node)
graph.add_edge(START, "BOT")
graph.add_conditional_edges("BOT", route_tools, {"tools": "tools", END: END})
graph.add_edge("tools", "BOT")

chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads():
    """Saved thread ids, oldest first (checkpoint.list yields newest checkpoints first)."""
    seen = []
    for ckpt in checkpoint.list(None):
        thread_id = ckpt.config["configurable"]["thread_id"]
        if thread_id not in seen:
            seen.append(thread_id)
    return list(reversed(seen))

