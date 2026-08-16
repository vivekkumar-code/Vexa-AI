from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(
    model="openrouter/free",
    max_tokens=100
)

result = model.invoke("Say hello")
print(result.content)
