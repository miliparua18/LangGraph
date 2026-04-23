import streamlit as st
from LangGraph_Backend import workflow
from langchain_core.messages import HumanMessage
import uuid

# ----------- Helper Functions -----------

def generate_thread_ID():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_ID()
    st.session_state['thread_id'] = thread_id
    st.session_state['messages'] = []

    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

    # default name
    st.session_state['thread_names'][thread_id] = "New Chat"

def add_threads(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = workflow.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get('messages', [])

# ----------- Streamlit Config -----------

st.set_page_config(page_title="LangGraph Chatbot")
st.title("🤖 AI Chatbot")

# ----------- Session State Init -----------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_ID()

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []

if "thread_names" not in st.session_state:
    st.session_state.thread_names = {}

# Add current thread
add_threads(st.session_state.thread_id)

# Ensure all threads have names
for t_id in st.session_state.chat_threads:
    if t_id not in st.session_state.thread_names:
        st.session_state.thread_names[t_id] = "New Chat"

# ----------- Sidebar -----------

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

for thread_id in st.session_state.chat_threads:
    name = st.session_state.thread_names.get(thread_id)

    if st.sidebar.button(name):
        st.session_state.thread_id = thread_id

        # Load messages from LangGraph
        messages = load_conversation(thread_id)

        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_messages.append({
                "role": role,
                "content": message.content
            })

        st.session_state.messages = temp_messages
        st.rerun()

# ----------- Display Messages -----------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------- User Input -----------

user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # ✅ Auto rename (first message only)
    if len(st.session_state.messages) == 1:
        st.session_state.thread_names[st.session_state.thread_id] = user_input[:25]

    CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}

    # ----------- Streaming Response -----------

    full_response = ""

    with st.chat_message("assistant"):
        for message_chunk, metadata in workflow.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages"
        ):
            if message_chunk.content:
                st.write(message_chunk.content)
                full_response += message_chunk.content

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })