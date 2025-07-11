# Llama Panel:

### Panel of Local LLMs to Deliver Superior, Consensus-Based Answers.

![Python Version](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Project Status](https://img.shields.io/badge/status-active-brightgreen)

---

## Go Beyond Single-Model Responses

Standard Large Language Models operate in isolation, limited by their training data and prone to singular viewpoints or biases. **Llama Panel** transforms this paradigm. It's a lightweight, powerful framework that implements a **Mixture-of-Models (MoM)** pattern, allowing you to run a local, agentic AI system where one expert model coordinates a panel of other LLMs to achieve a robust, multi-perspective consensus.

## Key Features

-   ✅ **Multi-Agent Collaboration:** Leverage the diverse strengths of multiple LLMs. Mitigate bias and generate more creative, well-rounded solutions by having a panel of specialists contribute to the final answer.

-   ✅ **Autonomous Orchestration:** An "Expert" agent analyzes user queries, formulates a plan, and independently decides which tools to use—whether to consult its panel, search the web, or read a specific webpage.

-   ✅ **Real-Time Web Access:** Overcome the knowledge cut-off problem. The expert agent can perform rich web searches (via DuckDuckGo) to gather up-to-the-minute information and read full webpage content for deep analysis.

-   ✅ **Fully Customizable:** You control the architecture. Define the expert model, assemble your panel with any Ollama-compatible models, and fine-tune the "creativity" (temperature) of each agent via simple command-line flags.

-   ✅ **High-Performance:** Asynchronous by design. All panel queries and network requests run concurrently, ensuring maximum throughput and minimal waiting time.

-   ✅ **Versatile & Scriptable:** Seamlessly switch between an interactive chat for conversational exploration and a single-question CLI mode perfect for scripting, automation, and integration into larger workflows.

## Architecture: The Reasoning Engine

Llama Panel operates on a dynamic, tool-based reasoning loop. The expert model is the core of this loop, making strategic decisions at each step to build towards a comprehensive answer.

```mermaid
graph TD
    subgraph "Llama Panel: Core Reasoning Loop"
        D[🧠 Expert Agent];
        E{Formulates Plan};
        F[🔎 search_web (DDG)];
        G[📖 get_webpage];
        I[📚 query_panel];
        H[✅ Synthesize Final Answer];

        D --> E;
        E --> |Request: Needs current facts| F;
        E --> |Request: Needs page content| G;
        E --> |Request: Needs diverse opinions| I;
        E --> |State: Sufficient data gathered| H;
        F --> |Data: Rich Search Results| D;
        G --> |Data: Full Webpage Text| D;
        I --> |Data: Panel Responses| D;
    end

    A[User / API] --> B{llama_panel.py};
    B --> |Initial Query| D;
    H --> A;
```

## Getting Started

### Prerequisites

1.  **Python 3.8+**
2.  **Ollama**: The Ollama service must be installed and running. Download from [ollama.com](https://ollama.com).
3.  **LLM Models**: Pull the models you intend to use. For the default configuration, you will need:
    ```bash
    ollama pull llama3
    ollama pull mistral
    ollama pull gemma:2b
    ```

### Installation

Clone or download `llama_panel.py` and install the required Python packages.

```bash
# Note the official package name for the search library is `ddgs`
pip install ollama termcolor ddgs httpx beautifulsoup4
```

Make the script executable for convenience:
```bash
chmod +x llama_panel.py
```

## Usage Guide

### Interactive Mode

For conversational sessions and complex, multi-step queries, run the script without arguments.
```bash
./llama_panel.py
```

### Command-Line Interface (CLI) Mode

For automation and scripting, provide the query as an argument. The final, clean answer is printed to `stdout`, while logs are sent to `stderr`.

```bash
# Get a single, high-quality answer
./llama_panel.py "Analyze the pros and cons of using WebAssembly as a server-side runtime."

# Pipe the result to a file
./llama_panel.py "Generate a boilerplate for a Python FastAPI project with JWT authentication." > fastapi_template.py
```

### Advanced Configuration

Tailor the agentic system to your specific needs using command-line flags.

**1. Set a Custom Expert Model:**
Use a more capable model like `llama3:70b` as the orchestrator for complex reasoning tasks.
```bash
./llama_panel.py --expert "llama3:70b:0.1" "Develop a multi-stage marketing plan for a new SaaS product."
```

**2. Assemble a Specialist Panel:**
Create a panel with a creative agent and a code-focused agent for a development task.
```bash
./llama_panel.py \
  --expert "llama3:8b:0.2" \
  --panel "mistral:1.2" "codellama:0.5" \
  "Brainstorm three novel features for a to-do list application and outline the code for the most promising one."
```

## Potential Use Cases

-   **Technical Research:** Ask complex questions about new technologies, and let the panel search, read documentation, and provide a synthesized summary.
-   **Content Creation:** Generate blog posts, marketing copy, or scripts by leveraging a creative panel for ideas and a factual expert for verification.
-   **Code Generation & Debugging:** Use a panel of coding models to suggest different approaches to a problem, review code for errors, and write boilerplate.
-   **Strategic Analysis:** Pose business or strategic questions and receive a multi-faceted analysis informed by web data and diverse AI perspectives.

## License

This project is licensed under the MIT License.