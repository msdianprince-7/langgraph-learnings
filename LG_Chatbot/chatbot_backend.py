from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import sqlite3

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-20b', reasoning_effort='low')


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: str  # sidebar name, saved in the database along with the chat


def chat_node(state: ChatState):
    response = model.invoke(state['messages'])
    return {'messages': [response]}


graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

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
