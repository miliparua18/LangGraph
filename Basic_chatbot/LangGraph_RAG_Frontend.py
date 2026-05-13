import streamlit as st
from LangGraph_RAG_Backend import chatbot, ingest_pdf, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# ---------------- Helper Functions ----------------

def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state.thread_id = thread_id
    st.session_state.messages = []

    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

    st.session_state.thread_names[thread_id] = "New Chat"


def load_conversation(thread_id):
    """Load conversation history from LangGraph memory"""

    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    messages = state.values.get("messages", [])

    formatted_messages = []

    for msg in messages:

        if isinstance(msg, HumanMessage):
            formatted_messages.append(
                {
                    "role": "user",
                    "content": msg.content
                }
            )

        elif isinstance(msg, AIMessage):
            formatted_messages.append(
                {
                    "role": "assistant",
                    "content": msg.content
                }
            )

    return formatted_messages


# ---------------- Streamlit Config ----------------

st.set_page_config(
    page_title="LangGraph PDF Chatbot",
    layout="wide"
)

st.title("🤖 PDF Chatbot")


# ---------------- Session State ----------------

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

if "thread_names" not in st.session_state:
    st.session_state.thread_names = {}

if "thread_id" not in st.session_state:

    if st.session_state.chat_threads:
        st.session_state.thread_id = st.session_state.chat_threads[0]

    else:
        st.session_state.thread_id = generate_thread_id()
        st.session_state.chat_threads.append(
            st.session_state.thread_id
        )

if "messages" not in st.session_state:
    st.session_state.messages = load_conversation(
        st.session_state.thread_id
    )

if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = {}


# ---------------- Sidebar ----------------

st.sidebar.title("LangGraph PDF Chatbot")

# New Chat Button
if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# ---------------- PDF Upload ----------------

uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

thread_id = st.session_state.thread_id

if uploaded_pdf:

    if thread_id not in st.session_state.uploaded_pdfs:
        st.session_state.uploaded_pdfs[thread_id] = []

    existing_files = st.session_state.uploaded_pdfs[thread_id]

    if uploaded_pdf.name not in existing_files:

        with st.sidebar.status(
            "Indexing PDF...",
            expanded=True
        ) as status:

            ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_id,
                filename=uploaded_pdf.name
            )

            existing_files.append(uploaded_pdf.name)

            status.update(
                label="✅ PDF Indexed",
                state="complete",
                expanded=False
            )

    else:
        st.sidebar.info(
            f"{uploaded_pdf.name} already uploaded."
        )

# Show Uploaded PDFs
if thread_id in st.session_state.uploaded_pdfs:

    pdfs = st.session_state.uploaded_pdfs[thread_id]

    if pdfs:
        st.sidebar.success(
            f"📄 {pdfs[-1]}"
        )

st.sidebar.divider()

# ---------------- Conversation History ----------------

st.sidebar.subheader("Recent Conversations")

for t_id in st.session_state.chat_threads:

    # Generate thread name
    if t_id not in st.session_state.thread_names:

        history = load_conversation(t_id)

        if history:
            name = history[0]["content"][:25] + "..."
        else:
            name = "Empty Chat"

        st.session_state.thread_names[t_id] = name

    else:
        name = st.session_state.thread_names[t_id]

    # Sidebar Button
    if st.sidebar.button(
        name,
        key=t_id,
        use_container_width=True
    ):

        st.session_state.thread_id = t_id

        st.session_state.messages = load_conversation(t_id)

        st.rerun()


# ---------------- Display Messages ----------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------- User Input ----------------

user_input = st.chat_input(
    "Ask anything about your PDF..."
)

if user_input:

    # Add User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Set Thread Name
    if len(st.session_state.messages) <= 2:

        st.session_state.thread_names[
            st.session_state.thread_id
        ] = user_input[:25] + "..."

    # Assistant Response
    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }

        # Stream Response
        for message_chunk, metadata in chatbot.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config,
            stream_mode="messages"
        ):

            if isinstance(message_chunk, AIMessage):

                if message_chunk.content:
                    full_response += message_chunk.content

                    placeholder.markdown(
                        full_response + "▌"
                    )

        placeholder.markdown(full_response)

    # Save Assistant Response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )