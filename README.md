# RAG-1: Retrieval-Augmented Generation Implementation

This repository demonstrates the implementation of Retrieval-Augmented Generation (RAG) using the LLaMA3 and GPT-2 models. Follow the instructions below to set up your environment and interact with the models.

---

## Table of Contents
- [Creating a Python Virtual Environment](#creating-a-python-virtual-environment)
- [Local LLM Setup and Interaction](#local-llm-setup-and-interaction)
- [Alternative LLM Installation](#alternative-llm-installation)

---

## Creating a Python Virtual Environment

To get started, you'll need to create a Python virtual environment. Follow the instructions for your operating system:

### For Windows:
```bash
python -m venv venv
```

### For macOS/Linux
```bash
python3 -m venv venv
```
After creating the virtual environment, activate it and install the required dependencies using the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Local LLM Setup and Interaction

1. **Download Ollama**  
   Visit the [Ollama GitHub Repository](https://github.com/ollama/ollama) and follow the installation instructions.

2. **Verify the Installation**  
   Run the following command to verify that Ollama is installed correctly:
   ```bash
   ollama run llama3.2
   ```

## Alternative LLM Installation
If you prefer to use GPT-2, ensure you have already installed the Hugging Face’s Transformers library from the `requirements.txt` file.

### To interact with GPT-2:
1. Create a Python file (e.g., `chat.py`) and load your preferred LLM (GPT-2) using the Transformers library.

2. Use the following command to run your script and ask GPT-2 your prompt:
   ```bash
   python file_name.py "Your_Prompt_here"
   ```


