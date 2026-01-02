“A platform that enables organizations to automatically fine-tune LLMs using domain-specific data with minimal human intervention.”
🏭 Agentic LLM Fine-Tuner
From Idea to Fine-Tuned Model in One Command.

This project is an autonomous AI pipeline built with LangGraph. It takes a high-level user goal (e.g., "Create a customer support bot for a sofa company"), scrapes the web 
for relevant knowledge, generates a synthetic training dataset, and automatically provisions a cloud GPU to fine-tune Llama 3 using Unsloth.

🧩 Architecture
The system operates as a State Machine with 4 distinct Agents:

🧠 Intake Agent (Local):

Model: Llama 3.1 (via Ollama).

Role: Analyzes the user's request and plans the dataset parameters (Topic, Style, Size).

🌐 Browser Agent (Cloud/Local):

Tools: Browserbase / DuckDuckGo.

Role: Searches the web and scrapes real-world text data to serve as the "knowledge base."

📝 Data Agent (Hybrid):

Model: GPT-4o (or Local Qwen 2.5-Coder).

Role: Cleans the scraped text and converts it into a strict JSONL dataset (instruction / output pairs).

⚡ Training Agent (Cloud):

Infrastructure: Modal (Serverless GPU).

Hardware: NVIDIA A10G.

Tech: Unsloth (LoRA + Quantization).

Role: Uploads the dataset, fine-tunes Llama 3, and pushes the adapter to Hugging Face.

🛠️ Tech Stack
Orchestration: LangGraph

Local Inference: Ollama

Cloud Compute: Modal

Fine-Tuning: Unsloth

Web Browsing: Browserbase / DuckDuckGo

Model Registry: Hugging Face

🚀 Setup & Installation
1. Prerequisites
Python 3.11 or 3.12 (Python 3.14 is currently unsupported by Modal).

Ollama installed and running locally.

A Hugging Face account (with a Write token).

A Modal account.
