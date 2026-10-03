import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot_backend import chatbot, generate_chat_title
import uuid


def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['chat_threads'].append(thread_id)
    st.session_state['message_history'] = []


def load_conversation(thread_id):
    # The checkpointer has the full chat for this thread; a thread with no messages yet has empty values
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', [])

    # Convert LangChain messages into the {'role', 'content'} format the UI uses
    history = []
    for message in messages:
        role = 'user' if isinstance(message, HumanMessage) else 'assistant'
        history.append({'role': role, 'content': message.content})
    return history


# Streamlit reruns this whole file on every message, so everything that must survive
# a rerun is kept in session_state. The bot's own memory is in the backend checkpointer.
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = [st.session_state['thread_id']]

if 'thread_names' not in st.session_state:
    st.session_state['thread_names'] = {}  # thread_id -> name shown in the sidebar

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

# Sidebar
st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('Start Chat'):
    reset_chat()
    st.rerun()  # rerun now so CONFIG and the page use the new thread straight away

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    # key=thread_id: a unique key per button, so Streamlit can tell them apart
    name = st.session_state['thread_names'].get(thread_id, 'New Chat')
    if st.sidebar.button(name, key=thread_id):
        st.session_state['thread_id'] = thread_id
        st.session_state['message_history'] = load_conversation(thread_id)
        st.rerun()

# Chat
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # stream_mode='messages' yields the LLM's reply token by token as (chunk, metadata)
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages',
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # Ask the LLM for a title after the first message (only once per chat).
    # The sidebar was already drawn above, so rerun once to show the new name.
    if st.session_state['thread_id'] not in st.session_state['thread_names']:
        st.session_state['thread_names'][st.session_state['thread_id']] = generate_chat_title(user_input)
        st.rerun()
