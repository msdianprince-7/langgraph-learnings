from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-20b', reasoning_effort='low')


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    response = model.invoke(state['messages'])
    return {'messages': [response]}


graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)


# A plain LLM call, outside the graph: it only labels the chat, it isn't part of the conversation
def generate_chat_title(first_message):
    prompt = ("Write a short title (max 5 words) for a chat that starts with this message. "
              f"Reply with only the title, no quotes.\n{first_message}")
    return model.invoke(prompt).content.strip()
