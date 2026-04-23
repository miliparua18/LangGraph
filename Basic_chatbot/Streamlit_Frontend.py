import streamlit as st
from LangGraph_Backend import workflow
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="LangGraph Chatbot")

st.title("🤖 AI Chatbot")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # backend call
    config = {"configurable": {"thread_id": "1"}}


    # Show AI response
    

    with st.chat_message("assistant"):
       ai_reply = st.write_stream(
           message_chunk.content for message_chunk, metadata in  workflow.stream(
               #initial state
               {"messages": [HumanMessage(content=user_input)]},
               #config
               config = {"configurable": {"thread_id": "1"}},
               stream_mode='messages' 
           )
        )
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})