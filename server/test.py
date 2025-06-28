import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")

print(GROQ_API_KEY)