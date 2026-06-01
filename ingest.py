import os
import uuid
import base64
import pdfplumber
import fitz  # PyMuPDF
import voyageai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
vc = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

def extract_text_chunks(pdf_path: str, chunk_size: int = 500) -> list[dict]:
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue
            words = text.split()
            current_chunk = []
            current_len = 0
            for word in words:
                current_chunk.append(word)
                current_len += len(word) + 1
                if current_len >= chunk_size:
                    chunks.append({"text": " ".join(current_chunk), "page": page_num + 1, "type": "text"})
                    current_chunk = []
                    current_len = 0
            if current_chunk:
                chunks.append({"text": " ".join(current_chunk), "page": page_num + 1, "type": "text"})
    return chunks

def extract_images(pdf_path: str) -> list[dict]:
    images = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            images.append({
                "image_b64": image_b64,
                "page": page_num + 1,
                "index": img_index,
                "type": "image",
                "ext": base_image["ext"]
            })
    return images

def embed_text_chunks(chunks: list[dict]) -> list[dict]:
    texts = [c["text"] for c in chunks]
    result = vc.multimodal_embed(
        inputs=[[t] for t in texts],
        model="voyage-multimodal-3",
        input_type="document"
    )
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = result.embeddings[i]
    return chunks

def embed_images(images: list[dict]) -> list[dict]:
    inputs = [[{"type": "image_base64", "image_base64": img["image_b64"]}] for img in images]
    result = vc.multimodal_embed(
        inputs=inputs,
        model="voyage-multimodal-3",
        input_type="document"
    )
    for i, img in enumerate(images):
        img["embedding"] = result.embeddings[i]
    return images

def upsert_to_pinecone(chunks: list[dict], images: list[dict], doc_id: str, user_id: str):
    text_vectors = []
    for chunk in chunks:
        text_vectors.append({
            "id": f"{doc_id}_text_{uuid.uuid4().hex[:8]}",
            "values": chunk["embedding"],
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"],
                "doc_id": doc_id,
                "user_id": user_id,
                "type": "text"
            }
        })
    if text_vectors:
        index.upsert(vectors=text_vectors, namespace="text")

    image_vectors = []
    for img in images:
        image_vectors.append({
            "id": f"{doc_id}_img_{uuid.uuid4().hex[:8]}",
            "values": img["embedding"],
            "metadata": {
                "image_b64": img["image_b64"],
                "page": img["page"],
                "doc_id": doc_id,
                "user_id": user_id,
                "type": "image",
                "ext": img["ext"]
            }
        })
    if image_vectors:
        index.upsert(vectors=image_vectors, namespace="images")

def ingest_pdf(pdf_path: str, doc_id: str = None, user_id: str = "anonymous"):
    if doc_id is None:
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    chunks = extract_text_chunks(pdf_path)
    images = extract_images(pdf_path)
    chunks = embed_text_chunks(chunks)
    if images:
        images = embed_images(images)
    upsert_to_pinecone(chunks, images, doc_id, user_id)

    return {"chunks": len(chunks), "images": len(images)}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
    else:
        ingest_pdf(sys.argv[1])
