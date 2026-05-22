import chromadb

client = chromadb.client()

collection = client.create_collection("policies")


