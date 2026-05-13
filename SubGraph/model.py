from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import os

def SubGraphmodel():
    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        streaming=True
    )

    model = ChatHuggingFace(llm = llm)

    return model

