from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re

# ======================
# MODEL SETUP
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

print("🧠 Student AI ready! Type 'exit' to quit.\n")

# ======================
# SIMPLE MATH SOLVER
# ======================

def try_solve_math(question):
    match = re.match(r"^\s*(\d+)\s*([\+\-\*/xX])\s*(\d+)\s*$", question)
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
# SESSION MEMORY
# ======================

conversation_history = []

# ======================
# CHAT LOOP
# ======================

while True:
    mode = input("Mode (math/science/theory/personal): ").lower()
    if mode == "exit":
        break

    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # ------------------
    # MATH TOOL CHECK
    # ------------------

    if mode == "math":
        result = try_solve_math(user_input)
        if result is not None:
            print(f"\nAssistant: Step-by-step calculation:\nThe result is {result}\n")
            continue

    # ------------------
    # MODE PROMPTS
    # ------------------

    if mode == "math":
        system_prompt = (
            "You are a math tutor. "
            "Solve problems step by step, verify calculations, "
            "and double-check the final answer."
        )

    elif mode == "science":
        system_prompt = (
            "You are a science tutor. "
            "Explain concepts clearly using simple language, "
            "step-by-step reasoning, and real-world examples."
        )

    elif mode == "theory":
        system_prompt = (
            "You are a study assistant for theory subjects. "
            "Define terms clearly, explain ideas, and give examples."
        )

    elif mode == "personal":
        system_prompt = (
            "You are a calm, supportive assistant. "
            "Give evidence-based advice, avoid speculation, "
            "and encourage healthy reasoning."
        )

    else:
        system_prompt = (
            "You are a helpful study assistant. "
            "Explain clearly and step by step."
        )

    # ------------------
    # BUILD MEMORY
    # ------------------

    history_text = ""
    for turn in conversation_history:
        history_text += (
            f"Question: {turn['question']}\n"
            f"Answer: {turn['answer']}\n\n"
        )

    # ------------------
    # FINAL PROMPT
    # ------------------

    prompt = (
        f"{system_prompt}\n\n"
        f"{history_text}"
        f"Question: {user_input}\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # ------------------
    # GENERATION
    # ------------------

    output = model.generate(
        **inputs,
        repetition_penalty=1.2,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nAssistant:", response, "\n")

    # ------------------
    # SAVE MEMORY
    # ------------------

    conversation_history.append({
        "question": user_input,
        "answer": response
    })

    if len(conversation_history) > 5:
        conversation_history.pop(0)
