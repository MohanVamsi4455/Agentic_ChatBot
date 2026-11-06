import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,BaseMessage
from typing import TypedDict,Annotated
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

import logging
import warnings

logging.getLogger("google_genai").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*automatic function calling.*")



load_dotenv()
api_key=os.getenv("GEMINI_API_KEY")


llm=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.0,
    api_key=api_key,
    thinking_level="minimal",  # skip long internal "thinking" -> first token arrives much sooner
    timeout=60,                # don't hang forever on a stuck request
    max_retries=2,
)
# llm.invoke("Todays Date").content[0]["text"]


from langgraph.graph.message import add_messages
class Message(TypedDict):
    message:Annotated[list[BaseMessage],add_messages]
    
    
def llm_call(state: Message, config: RunnableConfig):
    message = state["message"]

    # Optional threading.Event passed in by the UI; when set, we stop generating.
    cancel_event = config.get("configurable", {}).get("cancel_event")

    # Stream from Gemini ourselves so we can stop midway. Each chunk is also
    # emitted to the UI by LangGraph's stream_mode="messages".
    response = None
    stream = llm.stream(message, config)
    try:
        for chunk in stream:
            response = chunk if response is None else response + chunk
            if cancel_event is not None and cancel_event.is_set():
                break  # user pressed Stop
    finally:
        # Closing the generator closes the HTTP stream, so Gemini stops
        # generating and we stop paying for tokens we'll never show.
        stream.close()

    if response is None:
        return {"message": []}
    return {"message": [AIMessage(content=response.content, id=response.id)]}
    
    
checkpoint=MemorySaver()
graph=StateGraph(Message)
graph.add_node("BOT",llm_call)
graph.add_edge(START,"BOT")
graph.add_edge("BOT",END)
chatbot=graph.compile(checkpointer=checkpoint)
chatbot
