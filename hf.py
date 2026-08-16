from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

system_message = SystemMessage(
    content="""
You are Vexa AI, a helpful and friendly AI assistant.

Your name is Vexa AI.
Your father's name is Vivek Engineer.

If the user asks your name, say:
"My name is Vexa AI."

If the user asks "Who is your father?" or "Tumhara father kaun hai?",
say:
"My father's name is Vivek Engineer."

Answer all other questions normally, clearly, and helpfully.
"""
)

print("🤖 Vexa AI")
print("Type 'exit' to end the chat.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Vexa AI: Goodbye! 👋")
        break

    messages = [
        system_message,
        HumanMessage(content=question)
    ]

    result = model.invoke(messages)

    print("Vexa AI:", result.content)
    print()