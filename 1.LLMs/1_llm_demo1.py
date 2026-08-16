from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key="OPENROUTER_API_KEY"
)
result = llm.invoke("hay")
print(result.content)