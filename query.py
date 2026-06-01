import os
import voyageai
from pinecone import Pinecone
from openai import OpenAI
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
vc = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def embed_query(query: str) -> list[float]:
    result = vc.multimodal_embed(
        inputs=[[query]],
        model="voyage-multimodal-3",
        input_type="query"
    )
    return result.embeddings[0]

def retrieve(query_embedding: list[float], top_k: int = 3, user_id: Optional[str] = None, doc_id: Optional[str] = None) -> dict:
    filter_dict = {}
    if doc_id:
        filter_dict["doc_id"] = {"$eq": doc_id}
    elif user_id:
        filter_dict["user_id"] = {"$eq": user_id}

    kwargs = dict(vector=query_embedding, top_k=top_k, include_metadata=True)
    if filter_dict:
        kwargs["filter"] = filter_dict

    text_results = index.query(namespace="text", **kwargs)
    image_results = index.query(namespace="images", **kwargs)
    return {"text_matches": text_results.matches, "image_matches": image_results.matches}

def build_messages(query: str, text_matches: list, image_matches: list) -> list:
    context_parts = []
    for i, match in enumerate(text_matches):
        page = match.metadata.get("page", "?")
        text = match.metadata.get("text", "")
        context_parts.append(f"[Source {i+1} – Page {page}]\n{text}")
    context_text = "\n\n".join(context_parts)

    content = [
        {
            "type": "text",
            "text": f"""You are a helpful assistant answering questions about research documents.
Use the retrieved context below to answer the question. Always cite which source/page your answer comes from.

Retrieved text context:
{context_text}

Question: {query}"""
        }
    ]

    for match in image_matches:
        image_b64 = match.metadata.get("image_b64", "")
        ext = match.metadata.get("ext", "png")
        page = match.metadata.get("page", "?")
        if image_b64:
            content.append({"type": "text", "text": f"[Retrieved image from page {page}]:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{ext};base64,{image_b64}"}
            })

    return [{"role": "user", "content": content}]

def query_rag(query: str, user_id: Optional[str] = None, doc_id: Optional[str] = None) -> dict:
    query_embedding = embed_query(query)
    results = retrieve(query_embedding, user_id=user_id, doc_id=doc_id)
    text_matches = results["text_matches"]
    image_matches = results["image_matches"]

    messages = build_messages(query, text_matches, image_matches)
    response = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=1024
    )

    answer = response.choices[0].message.content
    sources = [
        {"page": m.metadata.get("page"), "doc_id": m.metadata.get("doc_id")}
        for m in text_matches
    ]

    return {
        "answer": answer,
        "sources": sources,
        "num_text_chunks": len(text_matches),
        "num_images": len(image_matches)
    }

if __name__ == "__main__":
    result = query_rag("What were Galileo's major contributions to astronomy?")
    print(result["answer"])
