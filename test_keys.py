import os
from dotenv import load_dotenv
load_dotenv()

# Test Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
print("Pinecone:", index.describe_index_stats())

# Test Voyage
import voyageai
vc = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
result = vc.embed(["hello world"], model="voyage-multimodal-3")
print("Voyage embedding dim:", len(result.embeddings[0]))

# Test Anthropic
import anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
msg = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=50, messages=[{"role":"user","content":"say hi"}])
print("Anthropic:", msg.content[0].text)