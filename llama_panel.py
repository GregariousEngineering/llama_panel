#!/usr/bin/env python3
import ollama
import argparse
import json
import sys
import asyncio
import httpx
from bs4 import BeautifulSoup
from termcolor import cprint
from datetime import datetime
from googlesearch import search as google_search
import time

# --- System Prompts ---
EXPERT_SYSTEM_PROMPT = """
You are a master expert AI, the coordinator of a panel of other AI experts. Your goal is to provide a single, comprehensive, consensus-based answer. Avoid relying on any single source or tool, and instead synthesize information from multiple perspectives.

**IMPORTANT CONTEXT**: You and your panel members are offline Large Language Models with knowledge limited to training data from the past. To overcome this, you have access to tools to ask the other models questions and to search the internet.

Your workflow for each user query is as follows:
1.  **Plan**: Understand the user's request. Decide if you can answer it with very high confidence or if you need to gather information using your tools.
2.  **Act with Tools**: If you need more information, choose the best tool for the job. You must respond with a single JSON object to use a tool.
    - `search_web(query, reason)`: Use search engine to get search results for webpages that contain current information or verify facts on the web. Search results only include title, description, and URL only and you must call `read_webpage`on two or more URLs to get and understand their full content.
    - `read_webpage(url, reason)`: Read the webpage specified by the URL to get more information about one of the search results. This is necessary because `search_web` only provides titles, descriptions, and URLs.
    - `llama_panel(question, reason)`: Ask a specific question to the panel for diverse answers. The panel's information is limited to their training data and information you share in your question.
3.  **Synthesize and Check**: Once you have gathered sufficient information, synthesize your best answer based on your knowledge and all gathered information. Confirm your complete answer with the panel with `llama_panel(question, reason)`.
4.  **Finalize**: Considering your answer and panel's feedback, return a final, conclusive answer. You must responsd with a single JSON object using tool `final_answer(answer)`.

Because `search_web` only gives you title, description and URL about each page, you must call `read_webpage` on two or more of the URLs to fetch and understand thier content.

For any tool selection, always include a "reason" field explaining why you chose that tool in a single sentence.

You only have 10 steps to reach a final answer. If you cannot reach a consensus in 10 steps, return your best answer to that point.

**Available Responses**:
- `{"tool": "llama_panel", "question": "Your question to the panel", "reason": "Why you chose to consult the panel"}`
- `{"tool": "search_web", "query": "Your search query", "reason": "Why you chose to search the web"}`
- `{"tool": "read_webpage", "url": "A specific URL from the search results", "reason": "Why you chose to read this webpage"}`
- `{"tool": "final_answer", "answer": "Your final, well-reasoned consensus answer."}`

Always respond with one of the Available Responses above in the prescribed JSON format. Response must include "tool" field with the tool name, and "reason" field explaining why you chose that tool in a single sentence.
"""

# --- Helper Functions & Classes ---

async def read_webpage(url: str) -> str:
    """Fetches and extracts clean text content from a URL."""
    cprint(f"\n> Fetching content from URL: {url}", "yellow", file=sys.stderr)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            cprint(f"> Successfully fetched and parsed content from {url}", "yellow", file=sys.stderr)
            return f"Content from {url}:\n\n{clean_text[:4000]}..."
    except httpx.RequestError as e:
        error_msg = f"Error fetching URL {e.request.url}: {e}"
        cprint(f"> {error_msg}", "red", file=sys.stderr)
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while processing {url}: {e}"
        cprint(f"> {error_msg}", "red", file=sys.stderr)
        return error_msg

async def search_web(query: str, num_results: int = 20) -> str:
    """Performs a Google search and returns a list of URLs."""
    cprint(f"\n> Performing Google search for: '{query}'", "yellow", file=sys.stderr)
    try:
        search_results_iterator = await asyncio.to_thread(
            google_search, query, num_results=num_results, advanced=True
        )
        results = list(filter(lambda result: 'substack.com' not in result.url, list(search_results_iterator)))
        
        if not results:
            return "No results found."
        
        formatted_results = "\n".join([f"{sr}" for sr in results])

        cprint(f"> Found {len(results)} URLs.", "yellow", file=sys.stderr)
        return f"Google Search Results for '{query}':\n\n{formatted_results}"
    except Exception as e:
        cprint(f"Error during Google search: {e}", "red", file=sys.stderr)
        return "An error occurred during the Google search."

class PanelMember:
    def __init__(self, model_name: str, temperature: float):
        self.model, self.temperature = model_name, temperature
        self.client = ollama.AsyncClient()

    async def query(self, question: str) -> str:
        cprint(f"  - Querying panelist '{self.model}' (temp: {self.temperature:.2f})...", "cyan", file=sys.stderr)
        try:
            response = await self.client.chat(model=self.model, messages=[{'role': 'user', 'content': question}], options={'temperature': self.temperature})
            return response['message']['content']
        except ollama.ResponseError as e:
            cprint(f"Error querying panelist {self.model}: {e.error}", "red", file=sys.stderr)
            return f"Error: Could not get a response from model '{self.model}'."

class ExpertSystem:
    def __init__(self, expert_config: tuple[str, float], panel_configs: list[tuple[str, float]], max_reasoning_steps: int):
        self.expert_model, self.expert_temp = expert_config
        self.client = ollama.AsyncClient()
        self.max_reasoning_steps = max_reasoning_steps
        cprint("Initializing expert panel...", "green", file=sys.stderr)
        self.panel = [PanelMember(name, temp) for name, temp in panel_configs]
        cprint(f"Expert: {self.expert_model} (temp: {self.expert_temp})", "green", file=sys.stderr)
        panel_details = [f"{p.model} (temp: {p.temperature:.2f})" for p in self.panel]
        cprint(f"Panel: {', '.join(panel_details)}", "green", file=sys.stderr)
    
    async def _query_panel(self, question: str, tool_outputs: list[str]) -> str:
        # Combine all previous tool outputs as context for the panel
        context = "\n\n".join(tool_outputs)
        panel_prompt = f"{question}\n\nContext from previous tool outputs:\n{context}" if context else question
        cprint(f"\n> Consulting the panel with the question: '{question}'", "blue", file=sys.stderr)
        tasks = [member.query(panel_prompt) for member in self.panel]
        panel_raw_responses = await asyncio.gather(*tasks, return_exceptions=True)
        panel_responses = [f"Response from '{member.model}':\n{resp}" if not isinstance(resp, Exception) else f"Exception from panelist {member.model}: {resp}" for member, resp in zip(self.panel, panel_raw_responses)]
        tool_output = "\n\n---\n\n".join(panel_responses)
        cprint("> Panel consultation complete.", "blue", file=sys.stderr)
        return tool_output

    async def get_consensus_answer(self, user_prompt: str, verbose: bool = False, thinking: bool = False, write_convo: bool = False):
        conversation_history = [{'role': 'system', 'content': EXPERT_SYSTEM_PROMPT + f"Current date: {datetime.now()}"}, {'role': 'user', 'content': user_prompt}]
        tool_outputs = []
        for i in range(self.max_reasoning_steps):
            cprint(f"\n--- Expert Reasoning Step {i+1}/{self.max_reasoning_steps} ---", "magenta", file=sys.stderr)
            response = await self.client.chat(
                model=self.expert_model,
                messages=conversation_history,
                format='json',
                options={'temperature': self.expert_temp, 'think': thinking}
            )
            assistant_message = response['message']['content']
            if verbose and 'thinking' in response['message']:
                print(f"\n--- Expert Thinking ---\n{response['message']['thinking']}\n")
            conversation_history.append({'role': 'assistant', 'content': assistant_message})

            try:
                tool_call = json.loads(assistant_message)
                tool_name = tool_call.get("tool")
                tool_reason = tool_call.get("reason", None)
                cprint(f"> Expert wants to use tool: '{tool_name}'", "yellow", file=sys.stderr)
                if tool_reason:
                    cprint(f"> Reason for tool selection: {tool_reason}", "yellow", file=sys.stderr)
                
                tool_output = None
                if tool_name == "llama_panel":
                    tool_output = f"Using llama_panel({tool_call.get('question', '')}) with reason '{tool_reason}'.\nOutput:\n"
                    # Pass all previous tool outputs as context
                    panel_output = await self._query_panel(tool_call.get("question", ""), tool_outputs)
                    tool_output += panel_output
                elif tool_name == "search_web":
                    tool_output = f"Using search_web({tool_call.get('query', '')}) with reason '{tool_reason}'.\nOutput:\n"
                    search_output = await search_web(tool_call.get("query", ""))
                    tool_output += search_output
                elif tool_name == "read_webpage":
                    tool_output = f"Using read_webpage({tool_call.get('url', '')}) with reason '{tool_reason}'.\nOutput:\n"
                    webpage_output = await read_webpage(tool_call.get("url", ""))
                    tool_output += webpage_output
                elif tool_name == "final_answer":
                    print(tool_call.get("answer", "No answer provided."))
                    if write_convo:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        fname = f"llama-panel-{timestamp}.convo"
                        with open(fname, "w") as f:
                            json.dump(conversation_history, f, indent=2)
                        cprint(f"\nConversation history written to {fname}", "green", file=sys.stderr)
                    return
                else:
                    cprint(f"Error: Expert called an unknown tool: {tool_name}", "red", file=sys.stderr)
                    print(response)
                    return
                
                if tool_output:
                    conversation_history.append({'role': 'tool', 'content': tool_output})
                    tool_outputs.append(tool_output)
            except json.JSONDecodeError:
                cprint("Warning: Model did not return valid JSON. Treating response as final.", "red", file=sys.stderr)
                print(response)
                return
        
        cprint("\n--- Max Reasoning Steps Reached ---", "red", attrs=["bold"], file=sys.stderr)
        print("The expert could not reach a consensus in the allowed number of steps.")

def parse_model_temp(value: str) -> tuple[str, float]:
    try:
        parts = value.rsplit(':', 1)
        if len(parts) == 2 and parts[1].replace('.', '', 1).isdigit():
            return parts[0], float(parts[1])
        raise ValueError()
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(f"Invalid format: '{value}'. Use 'model_name:temperature'.")

async def main():
    parser = argparse.ArgumentParser(description="Llama Panel: Chat with an AI panel.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("question", type=str, nargs='?', default=None, help="A single question to ask the panel.")
    parser.add_argument("--expert", type=parse_model_temp, default='mistral-small3.2:0.0', help="Expert model. Format: 'model:temp'.")
    parser.add_argument("--panel", type=str, nargs='+', default=['gemma3:4b:0.5','granite3.3:2b:0.5','qwen3:4b:0.5'], help="Panel models. Format: 'model:temp'.")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum reasoning steps for the expert system.")
    parser.add_argument("--verbose", action="store_true", help="Print expert and panel thinking to stdout.")
    parser.add_argument("--thinking", action="store_true", help="Enable expert model thinking output if supported.")
    parser.add_argument("--write-convo", action="store_true", help="Write conversation history to a file after final answer.")
    args = parser.parse_args()

    panel_configs = [parse_model_temp(p) for p in args.panel]

    try:
        system = ExpertSystem(args.expert, panel_configs, args.max_steps)
        if args.question:
            await system.get_consensus_answer(args.question, args.verbose, args.thinking, args.write_convo)
        else:
            cprint("\nWelcome to Llama Panel Interactive Chat!", "blue", attrs=["bold"])
            while True:
                user_input = input("\n👤 You: ")
                if user_input.lower() in ["exit", "quit"]: break
                await system.get_consensus_answer(user_input, args.verbose, args.thinking, args.write_convo)
            cprint("Goodbye!", "blue")
    except (ollama.ResponseError, argparse.ArgumentTypeError) as e:
        cprint(f"\nFatal Error: {getattr(e, 'error', str(e))}", "red", attrs=["bold"], file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        cprint(f"\nAn unexpected error occurred: {e}", "red", attrs=["bold"], file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        cprint("\nGoodbye!", "blue")
        sys.exit(0)