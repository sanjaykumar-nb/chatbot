import os
import json
import random
import string
import smtplib
import ssl
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from transformers import pipeline

# --- 1. SETUP ---
load_dotenv()
app = FastAPI(title="Sahayak AI Assistant")
PROFANITY_LIST = {"badword1", "exampleprofanity", "badword2"}

# In-memory storage for OTPs. For a real app, use a database.
otp_storage = {}

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.middleware("http")
async def profanity_filter_middleware(request: Request, call_next):
    if request.url.path == "/chat":
        try:
            body_bytes = await request.body()
            body_json = json.loads(body_bytes)
            user_text = body_json.get("text", "").lower()
            if any(word in user_text for word in PROFANITY_LIST):
                return JSONResponse(status_code=400, content={"detail": "Inappropriate language detected."})
        except Exception:
            pass
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request = Request(request.scope, receive)
    response = await call_next(request)
    return response

# --- 2. LOAD MODELS & VECTOR STORE AT STARTUP ---
print("Loading embeddings model and vector store...")
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    cache_folder='/data/models-cache' # Explicitly set cache folder for deployment
)
db = FAISS.load_local('vector_store/', embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={'k': 2})
print("Embeddings and vector store loaded.")

print("Setting up remote Chat LLM...")
# Create the base endpoint connection
llm_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.1,
    max_new_tokens=512
)
# Wrap the endpoint in the ChatHuggingFace class to handle formatting
llm_chat = ChatHuggingFace(llm=llm_endpoint)
print("Chat LLM loaded.")

print("Setting up summarization pipeline...")
summarizer = HuggingFacePipeline.from_model_id(
    model_id="facebook/bart-large-cnn",
    task="summarization",
    pipeline_kwargs={"max_length": 150, "min_length": 30, "no_repeat_ngram_size": 3},
)
print("Summarization pipeline ready.")

print("Setting up NER pipeline for keyword extraction...")
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")
keyword_extractor = HuggingFacePipeline(pipeline=ner_pipeline)
print("Keyword extraction pipeline ready.")


# --- 3. CREATE THE RAG CHAIN ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and respectful assistant for company employees. Answer the user's question based only on the following context. If the context doesn't contain the answer, say you don't have enough information."),
    ("human", "Context:\n{context}\n\nUser's Question:\n{question}"),
])
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} | prompt_template | llm_chat | StrOutputParser()
)

# --- 4. API ENDPOINTS ---
class Query(BaseModel):
    text: str

# Helper function to send email
def send_otp_email(receiver_email: str, otp: str):
    sender_email = os.getenv("SENDER_EMAIL")
    password = os.getenv("SENDER_PASSWORD")
    if not sender_email or not password:
        print("ERROR: SENDER_EMAIL or SENDER_PASSWORD environment variables not set.")
        return False

    message = f"""
    Subject: Your OTP for Sahayak Assistant

    Your One-Time Password is: {otp}
    This OTP is valid for 5 minutes.
    """
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
        print(f"OTP email sent to {receiver_email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

# Pydantic models for 2FA
class OtpRequest(BaseModel):
    email: EmailStr

class OtpVerify(BaseModel):
    email: EmailStr
    otp: str

@app.get("/")
def read_root():
    return {"message": "Sahayak AI Assistant API is running!"}

# Endpoint to request an OTP
@app.post("/request-otp")
def request_otp(data: OtpRequest):
    otp = ''.join(random.choices(string.digits, k=6))
    expiry = datetime.now() + timedelta(minutes=5)
    otp_storage[data.email] = {"otp": otp, "expiry": expiry}

    if send_otp_email(data.email, otp):
        return {"message": "OTP sent successfully to your email."}
    else:
        return JSONResponse(status_code=500, content={"message": "Failed to send OTP email."})

# Endpoint to verify an OTP
@app.post("/verify-otp")
def verify_otp(data: OtpVerify):
    stored_data = otp_storage.get(data.email)
    if not stored_data:
        return JSONResponse(status_code=400, content={"message": "OTP not requested for this email or has expired."})

    if datetime.now() > stored_data["expiry"]:
        del otp_storage[data.email]
        return JSONResponse(status_code=400, content={"message": "OTP has expired."})

    if stored_data["otp"] == data.otp:
        del otp_storage[data.email]
        return {"message": "OTP verified successfully. Access granted."}
    else:
        return JSONResponse(status_code=400, content={"message": "Invalid OTP."})

@app.post("/chat")
def chat(query: Query):
    # In a real app, you would protect this endpoint,
    # only allowing access after successful OTP verification.
    print(f"Received query: {query.text}")
    response = rag_chain.invoke(query.text)
    print(f"Generated response: {response}")
    return {"answer": response}

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    # This endpoint should also be protected in a real app.
    print(f"Received file: {file.filename}")
    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        summary = summarizer.invoke(text)
        entities = keyword_extractor.invoke(text)
        keywords = sorted(list(set([entity['word'] for entity in entities if entity['entity_group'] in ['ORG', "PER", 'LOC', 'MISC']])))
        return {"filename": file.filename, "summary": summary, "keywords": keywords}
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}