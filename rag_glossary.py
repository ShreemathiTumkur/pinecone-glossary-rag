import json
import openai
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load env variables
load_dotenv()
print("Loaded PINECONE_API_KEY:", os.getenv("PINECONE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_cloud = os.getenv("PINECONE_CLOUD")
pinecone_region = os.getenv("PINECONE_REGION")

# Initialize Pinecone client (Serverless)


index_name = "glossary-index"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region)
    )

# Connect to index
index = pc.Index(index_name)

# Load glossary terms
with open("glossary.json", "r") as f:
    glossary = json.load(f)

# Function to embed text
from openai import OpenAI

client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


# Upsert glossary into Pinecone
for item in glossary:
    term = item["term"]
    definition = item["definition"]
    vector = get_embedding(definition)
    index.upsert(vectors=[{
        "id": term,
        "values": vector,
        "metadata": {"definition": definition}
    }])

print("✅ Glossary embedded and indexed successfully!")

# Run a query
def search(query):
    query_vector = get_embedding(query)
    results = index.query(vector=query_vector, top_k=1, include_metadata=True)

    matches = results["matches"]
    if not matches:
        print("❌ No match found for query.")
        return

    match = matches[0]
    print(f"\n🔎 Query: {query}")
    print(f"✅ Closest match: {match['id']}")
    print(f"📚 Definition: {match['metadata']['definition']}")










    match = results["matches"][0]
    print(f"\n🔎 Query: {query}")
    print(f"✅ Closest match: {match['id']}")
    print(f"📚 Definition: {match['metadata']['definition']}")

# Example query
search("What is retrieval-augmented generation?")
