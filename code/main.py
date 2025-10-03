from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # Reverted to original HuggingFace embeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # Only for LLM
import os
import tempfile
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Question Answering API", description="API to answer questions based on an uploaded PDF document using LangChain and FAISS")

def load_llm_model(api_key: str):
    try:
        genai.configure(api_key=api_key)
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.2,
            top_p=0.95,
            convert_system_message_to_human=True
        )
        return model
    except Exception as e:
        logger.error(f"Error loading LLM model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load LLM model: {str(e)}")

def process_pdf(pdf_file: UploadFile):
    logger.info(f"Processing PDF: {pdf_file.filename}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.file.read())
        tmp_file_path = tmp_file.name

    try:
        pdf_loader = PyPDFLoader(tmp_file_path)
        pages = pdf_loader.load_and_split()
        logger.info(f"Loaded {len(pages)} pages from PDF")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        context = "\n\n".join(str(p.page_content) for p in pages)
        texts = text_splitter.split_text(context)
        logger.info(f"Created {len(texts)} text chunks")

        return texts
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        try:
            os.unlink(tmp_file_path)
            logger.info(f"Deleted temporary file: {tmp_file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {str(e)}")

@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    api_key: str = Form(...),
    pdf_file: UploadFile = File(...)
):
    logger.info(f"Received question: {question}")
    if not pdf_file.filename.lower().endswith(".pdf"):
        logger.error("Uploaded file is not a PDF")
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")

    try:
        texts = process_pdf(pdf_file)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_index = FAISS.from_texts(texts, embedding_model)
        logger.info("FAISS vector index created successfully")

        model = load_llm_model(api_key)

        template = """Use the following context to answer the question in a detailed manner.
                    If the context does not contain enough information, say "I don't know."
                    Provide examples or explanations if possible.
                    
                    Context:
                    {context}
                    
                    Question:
                    {question}
                    
                    Answer in detail:
                    """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=vector_index.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        logger.info("Running QA chain")
        result = qa_chain({"query": question})

        return {
            "answer": result["result"],
            "source_documents": [doc.page_content for doc in result.get("source_documents", [])]
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the PDF Question Answering API. Use the /ask endpoint to upload a PDF and submit questions."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000) 