import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot_backend import chatbot

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# Streamlit reruns this whole file on every message, so the chat shown on screen
# is kept in session_state. The bot's own memory is in the backend checkpointer.
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

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
