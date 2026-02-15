# ======================
# IMPORTS
# ======================

import re
import json
import os
import time
import requests

import faiss
from sentence_transformers import SentenceTransformer

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel


# ======================
# SEMANTIC RAG SETUP
# ======================

embedder = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("kb_index.faiss")

with open("kb_docs.json", "r", encoding="utf-8") as f:
    kb_documents = json.load(f)


# ======================
# LOAD MODEL (AUTO GPU/CPU)
# ======================

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_path = "student_ai_lora"

tokenizer = AutoTokenizer.from_pretrained(base_model)

if torch.cuda.is_available():
    print("🚀 CUDA detected. Loading 4-bit quantized model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )

else:
    print("⚠ CUDA not detected. Loading CPU model (slower)...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="cpu"
    )

model = PeftModel.from_pretrained(model, lora_path)
model.eval()

print("🧠 Student AI with RAG + Tool Routing ready. Type 'exit' to quit.\n")


# ======================
# MEMORY SYSTEM
# ======================

chat_history = []
MAX_HISTORY = 6


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
# WIKIPEDIA (REST + RETRY)
# ======================

def fetch_wikipedia(query, retries=3, timeout=5):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"

    for _ in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                return data.get("extract")
        except requests.exceptions.RequestException:
            time.sleep(1)

    return None


# ======================
# DUCKDUCKGO FALLBACK
# ======================

def fetch_duckduckgo(query, timeout=5):
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_redirect": 1,
        "no_html": 1
    }

    try:
        response = requests.get(url, params=params, timeout=timeout)
        data = response.json()
        return data.get("AbstractText")
    except requests.exceptions.RequestException:
        return None


# ======================
# SEMANTIC RETRIEVER (FAISS)
# ======================

def retrieve_context(query, top_k=2):
    try:
        query_vec = embedder.encode([query])
        _, indices = faiss_index.search(query_vec, top_k)

        retrieved = ""
        for idx in indices[0]:
            if idx < len(kb_documents):
                retrieved += kb_documents[idx] + "\n\n"

        return retrieved.strip()

    except Exception:
        return None


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
            print(f"\nAssistant:\n{result}\n")
            continue

    context = ""

    # ---------- KB ----------
    if tool == "kb":
        context = retrieve_context(user_input)

    # ---------- WIKI ----------
    elif tool == "wiki":
        wiki_data = fetch_wikipedia(user_input)

        if wiki_data:
            context = f"Wikipedia:\n{wiki_data}"
        else:
            duck_data = fetch_duckduckgo(user_input)
            if duck_data:
                context = f"DuckDuckGo:\n{duck_data}"

    # ---------- WEB ----------
    elif tool == "web":
        duck_data = fetch_duckduckgo(user_input)
        if duck_data:
            context = f"DuckDuckGo:\n{duck_data}"

    if not context:
        context = "Context is insufficient."

    # ---------- MEMORY ----------
    memory_context = ""
    for exchange in chat_history[-MAX_HISTORY:]:
        memory_context += f"User: {exchange['user']}\n"
        memory_context += f"Assistant: {exchange['assistant']}\n"

    # ---------- PROMPT ----------
    prompt = f"""
You are a student-focused AI assistant.

RULES:
- Use ONLY the provided CONTEXT
- Do NOT hallucinate
- If context is insufficient, say so clearly

CONTEXT:
{context}

CONVERSATION HISTORY:
{memory_context}

Question:
{user_input}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        repetition_penalty=1.2,
        do_sample=True
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Clean output (hide system prompt)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    print("\nAssistant:\n", response, "\n")

    # Save to memory
    chat_history.append({
        "user": user_input,
        "assistant": response
    })
