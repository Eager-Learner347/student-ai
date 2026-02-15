\# Student AI – Custom TinyLlama Assistant



\## Overview

Student AI is a custom-built AI assistant developed using TinyLlama and LoRA fine-tuning techniques. 

The project focuses on building a structured, tool-enhanced AI system with future integration of 

Retrieval-Augmented Generation (RAG) and ethical hacking modules.


## Hardware Support

- Automatically detects CUDA GPU
- Loads 4-bit quantized model if GPU available
- Falls back to CPU mode if no GPU detected



This project demonstrates practical implementation of:

\- Large Language Model fine-tuning

\- Parameter-Efficient Fine-Tuning (LoRA)

\- Output structure enforcement

\- Tool override mechanisms

\- AI system modular architecture



---



\## Features



\- Custom fine-tuned TinyLlama model

\- LoRA-based parameter-efficient training

\- Structured response formatting

\- Math tool override integration

\- Modular system design

\- Future RAG integration (in progress)

\- Ethical hacking + AI experimentation (planned phase)



---



\## Tech Stack



\- Python

\- PyTorch

\- HuggingFace Transformers

\- PEFT (LoRA)

\- TinyLlama

\- CUDA (GPU acceleration)



---



\## Project Structure



student-ai/

│

├── training/          # Model training scripts

├── lora/              # LoRA configuration and adapters

├── tools/             # Tool override implementations

├── rag/               # Retrieval-Augmented Generation (in progress)

├── student\_ai.py      # Main execution file

├── requirements.txt

└── README.md



---



\## Installation



1\. Clone the repository:

&nbsp;  git clone https://github.com/YOUR\_USERNAME/student-ai.git



2\. Navigate to project directory:

&nbsp;  cd student-ai



3\. Install dependencies:

&nbsp;  pip install -r requirements.txt



---



\## How It Works



1\. Loads TinyLlama base model

2\. Applies LoRA adapters

3\. Enforces structured output format

4\. Overrides specific tool behaviors (e.g., math operations)

5\. Generates responses based on user input



---



\## Future Improvements



\- Full RAG integration

\- Web retrieval capability

\- Advanced reasoning modules

\- Security-focused AI integration

\- Deployment via web interface



---



\## Author

Omar



---



\## License

MIT License



