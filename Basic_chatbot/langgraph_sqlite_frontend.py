import streamlit as st
from langGraph_sqlite_backend import workflow, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# ----------- Helper Functions -----------

def generate_thread_ID():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_ID()
    st.session_state.thread_id = thread_id
    st.session_state.messages = []
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)
    st.session_state.thread_names[thread_id] = "New Chat"

def load_conversation(thread_id):
    """Fetches history from LangGraph and formats it for Streamlit UI."""
    state = workflow.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get('messages', [])
    
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_messages.append({"role": "assistant", "content": msg.content})
    return formatted_messages

# ----------- Streamlit Config -----------

st.set_page_config(page_title="LangGraph Chatbot", layout="wide")
st.title("🤖 AI Chatbot")

# ----------- Session State Init -----------

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

if "thread_names" not in st.session_state:
    st.session_state.thread_names = {}

if "thread_id" not in st.session_state:
    # Default to the first existing thread or create a new one
    if st.session_state.chat_threads:
        st.session_state.thread_id = st.session_state.chat_threads[0]
    else:
        st.session_state.thread_id = generate_thread_ID()
        st.session_state.chat_threads.append(st.session_state.thread_id)

if "messages" not in st.session_state:
    st.session_state.messages = load_conversation(st.session_state.thread_id)

# ----------- Sidebar -----------

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Recent Conversations")

for t_id in st.session_state.chat_threads:
    # Determine display name
    if t_id not in st.session_state.thread_names:
        history = load_conversation(t_id)
        name = history[0]["content"][:25] + "..." if history else "Empty Chat"
        st.session_state.thread_names[t_id] = name
    else:
        name = st.session_state.thread_names[t_id]

    # Sidebar button for selecting thread
    if st.sidebar.button(name, key=t_id, use_container_width=True):
        st.session_state.thread_id = t_id
        st.session_state.messages = load_conversation(t_id)
        st.rerun()

# ----------- Display Messages -----------

# Display existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------- User Input -----------

user_input = st.chat_input("Type your message...")

if user_input:
    # 1. Add and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Update thread name if it's the first message
    if len(st.session_state.messages) <= 2: # User message + potentially one response
        st.session_state.thread_names[st.session_state.thread_id] = user_input[:25] + "..."

    # 3. Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # stream_mode="messages" yields message chunks
        for message_chunk, metadata in workflow.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        ):
            if message_chunk.content:
                full_response += message_chunk.content
                # Add a blinking cursor effect
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    # 4. Save final assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})