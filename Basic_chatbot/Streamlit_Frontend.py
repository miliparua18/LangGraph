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

    response = workflow.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    ai_reply = response["messages"][-1].content

    # Show AI response
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})

    with st.chat_message("assistant"):
        st.markdown(ai_reply)