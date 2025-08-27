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
from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings, HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# --- 1. SETUP ---
load_dotenv()
app = FastAPI(title="Sahayak AI Assistant")
PROFANITY_LIST = {"badword1", "exampleprofanity", "badword2"}
otp_storage = {}
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

# --- 2. LOAD MODELS & VECTOR STORE AT STARTUP (NOW ULTRA-LIGHTWEIGHT) ---
print("Setting up API-based embeddings...")
embeddings = HuggingFaceInferenceAPIEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

print("Loading vector store...")
db = FAISS.load_local('vector_store/', embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={'k': 2})
print("Vector store loaded.")

print("Setting up remote Chat LLM...")
llm_endpoint = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.1, max_new_tokens=512)
llm_chat = ChatHuggingFace(llm=llm_endpoint)
print("Chat LLM loaded.")

print("Setting up summarization pipeline...")
summarizer = HuggingFaceEndpoint(repo_id="facebook/bart-large-cnn", task="summarization")
print("Summarization pipeline ready.")

print("Setting up NER pipeline for keyword extraction...")
keyword_extractor = HuggingFaceEndpoint(repo_id="dslim/bert-base-NER", task="token-classification")
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

def send_otp_email(receiver_email: str, otp: str):
    sender_email = os.getenv("SENDER_EMAIL")
    password = os.getenv("SENDER_PASSWORD")
    if not sender_email or not password:
        print("ERROR: SENDER_EMAIL or SENDER_PASSWORD environment variables not set.")
        return False
    message = f"Subject: Your OTP for Sahayak Assistant\n\nYour One-Time Password is: {otp}"
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

class OtpRequest(BaseModel):
    email: EmailStr

class OtpVerify(BaseModel):
    email: EmailStr
    otp: str

@app.get("/")
def read_root():
    return {"message": "Sahayak AI Assistant API is running!"}

@app.post("/request-otp")
def request_otp(data: OtpRequest):
    otp = ''.join(random.choices(string.digits, k=6))
    expiry = datetime.now() + timedelta(minutes=5)
    otp_storage[data.email] = {"otp": otp, "expiry": expiry}
    if send_otp_email(data.email, otp):
        return {"message": "OTP sent successfully."}
    else:
        return JSONResponse(status_code=500, content={"message": "Failed to send OTP email."})

@app.post("/verify-otp")
def verify_otp(data: OtpVerify):
    stored_data = otp_storage.get(data.email)
    if not stored_data or datetime.now() > stored_data["expiry"] or stored_data["otp"] != data.otp:
        if stored_data and datetime.now() > stored_data["expiry"]:
             del otp_storage[data.email]
        return JSONResponse(status_code=400, content={"message": "Invalid or expired OTP."})

    del otp_storage[data.email]
    return {"message": "OTP verified successfully. Access granted."}

@app.post("/chat")
def chat(query: Query):
    print(f"Received query: {query.text}")
    response = rag_chain.invoke(query.text)
    print(f"Generated response: {response}")
    return {"answer": response}

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        
        # We need to manually parse the JSON output from these endpoints now
        summary_result = summarizer.invoke({"inputs": text})
        summary = summary_result[0].get('summary_text', "Could not generate summary.")
        
        entities = keyword_extractor.invoke({"inputs": text})
        keywords = sorted(list(set([entity['word'] for entity in entities if entity.get('entity_group') in ['ORG', "PER", 'LOC', 'MISC']])))
        
        return {"filename": file.filename, "summary": summary, "keywords": keywords}
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}