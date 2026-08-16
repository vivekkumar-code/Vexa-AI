from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

result = client.embeddings.create(
    model="nvidia/nemotron-3-embed-1b:free",
    input="Delhi is the capital of India",
    encoding_format="float"
)

print(len(result.data[0].embedding))

