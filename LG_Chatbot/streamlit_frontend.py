import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from chatbot_backend import chatbot, generate_chat_title, retrieve_all_threads
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
    # Skip tool results and the empty AI messages that only asked for a tool
    history = []
    for message in messages:
        if isinstance(message, ToolMessage) or not message.content:
            continue
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
    # Old chats come from the database (newest first, so reverse to oldest first), then the new empty chat
    st.session_state['chat_threads'] = retrieve_all_threads()[::-1] + [st.session_state['thread_id']]

if 'thread_names' not in st.session_state:
    # thread_id -> name shown in the sidebar, read back from each saved chat's 'title'
    st.session_state['thread_names'] = {}
    for thread_id in st.session_state['chat_threads']:
        title = chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values.get('title')
        if title:
            st.session_state['thread_names'][thread_id] = title

CONFIG = {
    'configurable': {'thread_id': st.session_state['thread_id']},  # checkpointer: which chat to load/save
    'metadata': {'thread_id': st.session_state['thread_id']},      # LangSmith: groups every turn of a chat into one Thread
    'run_name': 'chat_turn',                                       # LangSmith: name of each trace
}

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

    # stream_mode='messages' yields every message as it's produced, as (chunk, metadata):
    # the LLM's tokens (AIMessage chunks) AND tool results (ToolMessage).
    with st.chat_message('assistant'):
        status_holder = {'box': None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages',
            ):
                # A tool ran: show a status box instead of printing the raw tool output
                if isinstance(message_chunk, ToolMessage):
                    tool_name = message_chunk.name
                    if status_holder['box'] is None:
                        status_holder['box'] = st.status(f'🔧 Using `{tool_name}` …', expanded=True)
                    else:
                        status_holder['box'].update(label=f'🔧 Using `{tool_name}` …', state='running')

                # Only the LLM's text goes into the reply
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder['box'] is not None:
            status_holder['box'].update(label='✅ Tool finished', state='complete', expanded=False)

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # Ask the LLM for a title after the first message (only once per chat).
    # The sidebar was already drawn above, so rerun once to show the new name.
    if st.session_state['thread_id'] not in st.session_state['thread_names']:
        title = generate_chat_title(user_input)
        st.session_state['thread_names'][st.session_state['thread_id']] = title
        chatbot.update_state(CONFIG, {'title': title})  # also save it in the database
        st.rerun()
