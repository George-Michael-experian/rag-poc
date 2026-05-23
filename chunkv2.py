from chunk import chunk_document
import chromadb

client  = chromadb.Client()
collection = client.create_collection("chunk_docs")
chunks = chunk_document("Large text...", chunk_size=100, overlap=20)

for i, chunk in enumerate(chunks):
    collection.add(
        documents = [chunk],
        ids = [f"chunk_{i}"]
    )

query = "What is the main topic of the text?"
results = collection.query(query_texts= [query], n_results=3)

for result in results['documents'][0]:
    print(f"result: {result}")
