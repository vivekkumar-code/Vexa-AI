from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenRouter(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    )
result=model.invoke("write five line idea on bird")
print(result.content)
