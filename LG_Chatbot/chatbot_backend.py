from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import sys
import threading
import queue
import aiosqlite
import yfinance as yf

load_dotenv()

# ---------- One event loop for all async work ----------
# Streamlit is not async, so a single asyncio loop runs forever in a background thread.
# Every async call (graph, checkpointer, later MCP tools) is sent to this loop.
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()


# Run a coroutine on the background loop and wait for its result
def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


model = ChatGroq(model='openai/gpt-oss-20b', reasoning_effort='low')

# ---------- Tools ----------
search_tool = DuckDuckGoSearchRun()  # ready-made LangChain tool: web search, no API key


@tool
def get_stock_price(symbol: str) -> str:
    """Get the latest stock price for a ticker symbol, e.g. AAPL, TSLA, or INFY.NS for NSE India."""
    price = yf.Ticker(symbol).fast_info['last_price']
    return f'{symbol.upper()}: {price:.2f}'


# The calculator now lives in its own MCP server. The client starts it as a subprocess
# (same Python as this app) and turns its MCP tools into normal LangChain tools.
mcp_client = MultiServerMCPClient({
    'calculator': {
        'transport': 'stdio',
        'command': sys.executable,
        'args': [str(Path(__file__).parent / 'calculator_mcp_server.py')],
    },
})
mcp_tools = run_async(mcp_client.get_tools())  # async-only, so it runs on the background loop

tools = [search_tool, get_stock_price, *mcp_tools]
model_with_tools = model.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: str  # sidebar name, saved in the database along with the chat


async def chat_node(state: ChatState):
    response = await model_with_tools.ainvoke(state['messages'])
    return {'messages': [response]}


graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', ToolNode(tools))  # sync tools are fine here: ToolNode runs them in a thread

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)  # tool call -> 'tools', else -> END
graph.add_edge('tools', 'chat_node')


# The async checkpointer must be created inside the loop it will be used on
async def _make_checkpointer():
    conn = await aiosqlite.connect('chatbot.db')
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_make_checkpointer())
chatbot = graph.compile(checkpointer=checkpointer)


# ---------- Sync helpers for the Streamlit frontend ----------
# Stream (message_chunk, metadata) pairs from the async graph as a normal generator
def stream_reply(user_input, config):
    q = queue.Queue()

    async def produce():
        try:
            async for item in chatbot.astream(
                {'messages': [HumanMessage(content=user_input)]}, config=config, stream_mode='messages'
            ):
                q.put(item)
        finally:
            q.put(None)  # tells the generator the stream is over

    future = asyncio.run_coroutine_threadsafe(produce(), _loop)
    while (item := q.get()) is not None:
        yield item
    future.result()  # re-raises any error from the graph


# The saved state of one chat (messages, title); empty dict for a new chat
def get_thread_values(thread_id):
    state = run_async(chatbot.aget_state({'configurable': {'thread_id': thread_id}}))
    return state.values


def save_title(config, title):
    run_async(chatbot.aupdate_state(config, {'title': title}))


# A plain LLM call, outside the graph: it only labels the chat, it isn't part of the conversation
def generate_chat_title(first_message):
    prompt = ("Write a short title (max 5 words) for a chat that starts with this message. "
              f"Reply with only the title, no quotes.\n{first_message}")
    response = run_async(model.ainvoke(prompt, config={'run_name': 'generate_chat_title'}))
    return response.content.strip()


# Every checkpoint in the database belongs to a thread, so collect the unique thread ids
def retrieve_all_threads():
    async def collect():
        all_threads = []
        async for checkpoint in checkpointer.alist(None):
            thread_id = checkpoint.config['configurable']['thread_id']
            if thread_id not in all_threads:
                all_threads.append(thread_id)
        return all_threads

    return run_async(collect())
