from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from typing import TypedDict, Annotated
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START,END
from langgraph.prebuilt import ToolNode, tools_condition
import os
from langsmith import traceable

load_dotenv()

llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Pro",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        streaming=True
    )

model = ChatHuggingFace(llm = llm)

#Load the PDF
loader = PyPDFLoader('Data/ML_Introduction_1000_Pages.pdf')
docs = loader.load()


#split the docs
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunk = splitter.split_documents(docs)

# convert text to vector
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = Chroma.from_documents(chunk, embeddings)

#create retriever
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})


@tool
def rag_tool(query):
    """
    Retrieve relevant information from the pdf documnets.
    use this tool when user asks factual/conceptual questions
    that might be answered from the stored documnets.
    """
    result = retriever.invoke(query)
    context = [docs.page_content for docs in result]
    metadata = [docs.metadata for docs in result]
    return{
        'query': query,
        'context' : context,
        'metadata': metadata
    }

tools = [rag_tool]
llm_with_tool = model.bind_tools(tools)


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

@traceable(name="pdf_rag")
def chat_node(state: ChatState):
    messages = state['messages']
    result = llm_with_tool.invoke(messages)
    return {"messages": [result]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node) 

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("chat_node", END )

chatbot = graph.compile()


result = chatbot.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "using the pdf notes , explain K-Nearest Neighbors"
                )
            )
        ]
    }
)

print(result['messages'][-1].content)

