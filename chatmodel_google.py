from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature = 0.7,
    max_output_tokens=500
)
result=model.invoke("uttar pradesh ka arthik vishesta in english")
print(result.content)