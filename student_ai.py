# ======================
# IMPORTS
# ======================

import re
import json
import os
import wikipedia
import faiss
from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()


# ======================
# SEMANTIC RAG SETUP
# ======================

embedder = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("kb_index.faiss")

with open("kb_docs.json", "r", encoding="utf-8") as f:
    kb_documents = json.load(f)


# ======================
# LOAD MODEL
# ======================

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_path = "student_ai_lora"

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, lora_path)
model.eval()

print("🧠 Student AI with RAG + Tool Routing ready. Type 'exit' to quit.\n")


# ======================
# HARD MATH OVERRIDE
# ======================

def solve_math(question):
    match = re.search(r"(\d+)\s*([\+\-\*/xX])\s*(\d+)", question)
    if not match:
        return None

    a = int(match.group(1))
    op = match.group(2)
    b = int(match.group(3))

    if op in ["*", "x", "X"]:
        return a * b
    elif op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "/":
        return a / b


# ======================
# TOOL ROUTER
# ======================

def route_tool(query):
    q = query.lower()

    if re.search(r"\d+\s*[\+\-\*/xX]\s*\d+", q):
        return "math"

    if any(word in q for word in ["latest", "current", "recent", "news", "update"]):
        return "web"

    if q.startswith(("what is", "who is", "define")):
        return "wiki"

    if any(word in q for word in ["why", "how", "explain", "describe"]):
        return "kb"

    return "llm"


# ======================
# WIKIPEDIA FETCH
# ======================

def fetch_wikipedia(query):
    try:
        return wikipedia.summary(query, sentences=4)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple meanings found: {e.options[:3]}"
    except wikipedia.exceptions.PageError:
        return None


# ======================
# WEB SEARCH
# ======================

def web_search(query):
    params = {
        "q": query,
        "api_key": os.getenv("SERPAPI_KEY"),
        "engine": "google",
        "num": 3
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    snippets = ""
    for res in results.get("organic_results", []):
        if "snippet" in res:
            snippets += res["snippet"] + "\n"

    return snippets.strip()


# ======================
# SEMANTIC RETRIEVER
# ======================

def retrieve_context(query, top_k=2):
    query_vec = embedder.encode([query])
    _, indices = faiss_index.search(query_vec, top_k)

    retrieved = ""
    for idx in indices[0]:
        if idx < len(kb_documents):
            retrieved += kb_documents[idx] + "\n\n"

    return retrieved.strip()


# ======================
# CHAT LOOP
# ======================

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    tool = route_tool(user_input)

    # ---------- MATH ----------
    if tool == "math":
        result = solve_math(user_input)
        if result is not None:
            print(
                f"\nAssistant:\n"
                f"Given:\n{user_input}\n\n"
                f"Steps:\nPerform the calculation.\n\n"
                f"Check:\nRecalculate to verify.\n\n"
                f"Final Answer:\n{result}\n"
            )
            continue

    # ---------- RETRIEVAL ----------
    context = ""

    if tool == "kb":
        context = retrieve_context(user_input)

    elif tool == "wiki":
        wiki_data = fetch_wikipedia(user_input)
        if wiki_data:
            context = f"Wikipedia:\n{wiki_data}"

    elif tool == "web":
        web_data = web_search(user_input)
        if web_data:
            context = f"Web Search Results:\n{web_data}"

    if not context:
        context = "Context is insufficient."

    # ---------- PROMPT ----------
    prompt = f"""
You are a student-focused AI assistant.

RULES:
- Use ONLY the provided CONTEXT
- Do NOT hallucinate
- If context is insufficient, say so clearly

CONTEXT:
{context}

FORMAT:
Given:
Steps:
Check:
Final Answer:

Question:
{user_input}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.2,
        repetition_penalty=1.2,
        do_sample=True
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nAssistant:\n", response, "\n")
