from sentence_transformers import SentenceTransformer
import faiss
import os
import json

DATA_DIR = "knowledge_base"
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

documents = []
doc_names = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)
            doc_names.append(file)

embeddings = model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "kb_index.faiss")

with open("kb_docs.json", "w", encoding="utf-8") as f:
    json.dump(documents, f)

print("✅ Embeddings built and stored.")
