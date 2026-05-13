from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os

def huggingface_model():
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Pro",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        streaming=True
    )

    model = ChatHuggingFace(llm=llm)
    return model