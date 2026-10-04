from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import sqlite3
import yfinance as yf

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-20b', reasoning_effort='low')

# ---------- Tools ----------
search_tool = DuckDuckGoSearchRun()  # ready-made LangChain tool: web search, no API key


@tool
def calculator(first_num: float, second_num: float, operation: str) -> str:
    """Do basic arithmetic on two numbers. operation is one of: add, sub, mul, div."""
    if operation == 'add':
        return str(first_num + second_num)
    elif operation == 'sub':
        return str(first_num - second_num)
    elif operation == 'mul':
        return str(first_num * second_num)
    elif operation == 'div':
        return str(first_num / second_num) if second_num != 0 else 'Division by zero is not allowed'
    return f'Unsupported operation: {operation}'


@tool
def get_stock_price(symbol: str) -> str:
    """Get the latest stock price for a ticker symbol, e.g. AAPL, TSLA, or INFY.NS for NSE India."""
    price = yf.Ticker(symbol).fast_info['last_price']
    return f'{symbol.upper()}: {price:.2f}'


tools = [search_tool, calculator, get_stock_price]
model_with_tools = model.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: str  # sidebar name, saved in the database along with the chat


def chat_node(state: ChatState):
    response = model_with_tools.invoke(state['messages'])
    return {'messages': [response]}


graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', ToolNode(tools))

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)  # tool call -> 'tools', else -> END
graph.add_edge('tools', 'chat_node')

# check_same_thread=False: Streamlit runs each rerun in a different thread, all sharing this one connection
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
chatbot = graph.compile(checkpointer=checkpointer)


# A plain LLM call, outside the graph: it only labels the chat, it isn't part of the conversation
def generate_chat_title(first_message):
    prompt = ("Write a short title (max 5 words) for a chat that starts with this message. "
              f"Reply with only the title, no quotes.\n{first_message}")
    return model.invoke(prompt, config={'run_name': 'generate_chat_title'}).content.strip()


# Every checkpoint in the database belongs to a thread, so collect the unique thread ids
def retrieve_all_threads():
    all_threads = []
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        if thread_id not in all_threads:
            all_threads.append(thread_id)
    return all_threads
