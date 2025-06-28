from fastapi import FastAPI,UploadFile,File,Form,Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from modules.load_vectorstore import load_vectorstore
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from logger import logger
import os

app=FastAPI(title="RagBot2.0")

# allow frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def catch_exception_middleware(request:Request,call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("UNHANDLED EXCEPTION")
        return JSONResponse(status_code=500,content={"error":str(exc)})
    
@app.post("/upload_pdfs/")
async def upload_pdfs(files:List[UploadFile]=File(...)):
    try:
        logger.info(f"recieved {len(files)} files")
        load_vectorstore(files)
        logger.info("documents added to chroma")
        return {"message":"Files processed and vectorstore updated"}
    except Exception as e:
        logger.exception("Error during pdf upload")
        return JSONResponse(status_code=500,content={"error":str(e)})


# @app.post("/ask/")
# async def ask_quyestion(question:str=Form(...)):
#     try:
#         logger.info("fuser query:{question}")
#         from langchain.vectorstores import Chroma
#         from langchain.embeddings import HuggingFaceBgeEmbeddings
#         from modules.load_vectorstore import PERSIST_DIR

#         vectorstore=Chroma(
#             persist_directory=PERSIST_DIR,
#             embedding_function=HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L12-v2")
#         )
#         chain=get_llm_chain(vectorstore)
#         result=query_chain(chain,question)
#         logger.info("query successfull")
#         return result
#     except Exception as e:
#         logger.exception("error processing question")
#         return JSONResponse(status_code=500,content={"error":str(e)})

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        from pinecone import Pinecone
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_core.documents import Document
        from langchain.schema import BaseRetriever
        from typing import List, Optional
        from pydantic import Field
        from modules.llm import get_llm_chain
        from modules.query_handlers import query_chain
        import os

        # 1. Pinecone + Embedding setup
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
        embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # 2. Embed the question
        embedded_query = embed_model.embed_query(question)

        # 3. Query Pinecone
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        # 4. Convert to LangChain Documents
        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            ) for match in res["matches"]
        ]

        # 5. Pydantic-compliant retriever subclass
        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

        retriever = SimpleRetriever(docs)

        # 6. LLM + RetrievalQA chain
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/test")
async def test():
    return {"message":"Testing successfull..."}