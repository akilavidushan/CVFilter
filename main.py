import os, tempfile, zipfile, shutil
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import fitz
from docx import Document
import openai

# --- FastAPI setup ---
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[  
        "*",
    ],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=[
        "Content-Type", 
        "Accept", 
        "Origin",   
        "Referer",   
        "User-Agent"   
    ]
)

# --- Azure config from environment variables ---
AZURE_STORAGE_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER = "filtered"
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Setup clients
blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONN)

openai.api_key = AZURE_OPENAI_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_type = "azure"
openai.api_version = "2024-02-15-preview"


@app.post("/filter")
async def filter_cvs(cvZip: UploadFile = File(...), prompt: str = Form(...)):
    with tempfile.TemporaryDirectory() as tmp:
        # Save uploaded ZIP
        zip_path = os.path.join(tmp, cvZip.filename)
        with open(zip_path, "wb") as f:
            f.write(await cvZip.read())

        # Extract
        extract_path = os.path.join(tmp, "extracted")
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        selected_files = []
        total = 0

        # Loop through extracted files
        for file_name in os.listdir(extract_path):
            file_path = os.path.join(extract_path, file_name)
            if not (file_name.endswith(".pdf") or file_name.endswith(".docx")):
                continue

            # Extract text
            if file_name.endswith(".pdf"):
                doc = fitz.open(file_path)
                text = "\n".join(page.get_text() for page in doc)
            else:
                doc = Document(file_path)
                text = "\n".join(p.text for p in doc.paragraphs)

            total += 1

            # Query GPT-4o via Azure OpenAI
            response = openai.ChatCompletion.create(
                engine=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a recruitment assistant. Answer only yes or no."},
                    {"role": "user", "content": f"Prompt: {prompt}"},
                    {"role": "user", "content": f"CV:\n{text[:8000]}"}
                ],
                temperature=0
            )

            reply = response.choices[0].message["content"]
            if "yes" in reply.lower():
                selected_files.append(file_path)

        # Zip selected files
        filtered_name = f"filtered_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.zip"
        filtered_path = os.path.join(tmp, filtered_name)
        with zipfile.ZipFile(filtered_path, "w") as zipf:
            for file in selected_files:
                zipf.write(file, arcname=os.path.basename(file))

        # Upload to Azure Blob
        blob = blob_service.get_blob_client(container=AZURE_CONTAINER, blob=filtered_name)
        with open(filtered_path, "rb") as data:
            blob.upload_blob(data)

        # Generate download URL with SAS token
        sas_token = generate_blob_sas(
            account_name=blob.account_name,
            container_name=AZURE_CONTAINER,
            blob_name=filtered_name,
            account_key=blob_service.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=2)
        )

        download_url = f"https://{blob.account_name}.blob.core.windows.net/{AZURE_CONTAINER}/{filtered_name}?{sas_token}"

        return JSONResponse({
            "download_url": download_url,
            "passed": len(selected_files),
            "total": total
        })
