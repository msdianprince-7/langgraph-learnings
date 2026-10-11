# LangGraph for Beginners

## What Is LangGraph?

LangGraph is a lightweight, graph‑oriented framework designed to simplify the construction of conversational AI pipelines. It treats a dialogue system as a directed graph where each node represents a distinct processing step—such as language understanding, retrieval, or generation—and edges define the flow of data between them. By visualizing and configuring interactions as a graph, developers can easily compose, test, and iterate on complex conversational flows without wrestling with low‑level orchestration code. This approach not only accelerates prototyping but also promotes modularity, reusability, and clearer debugging, making LangGraph a valuable tool for anyone looking to build robust, scalable chatbot experiences.

## Getting Started: Install and Set Up

Below is a step‑by‑step guide to installing LangGraph, preparing a clean Python environment, and creating the skeleton for your first LangGraph project.

### 1. Create a Project Directory

```bash
mkdir my_langgraph_project
cd my_langgraph_project
```

### 2. Set Up a Virtual Environment

Using `venv` (built‑in) keeps dependencies isolated.

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install LangGraph

```bash
pip install langgraph
```

Verify the installation:

```bash
python -c "import langgraph; print(langgraph.__version__)"
```

You should see a version number printed.

### 4. Project Structure

Organize the repository for clarity and scalability.

```
my_langgraph_project/
├── .venv/                # virtual environment (ignored by .gitignore)
├── src/
│   ├── __init__.py
│   ├── graph.py          # your graph definitions
│   └── utils.py          # helper functions
├── data/
│   └── sample_data.json
├── tests/
│   └── test_graph.py
├── requirements.txt
└── README.md
```

- **`src/`** – core code.
- **`data/`** – any static files or sample data.
- **`tests/`** – unit tests for your graph logic.

### 5. Add a Simple Graph

Create `src/graph.py`:

```python
from langgraph import Graph, Node, Edge

class MyGraph(Graph):
    def __init__(self):
        super().__init__()

        # Define nodes
        self.add_node(Node("start", lambda ctx: print("Start node executed")))
        self.add_node(Node("process", lambda ctx: print("Processing...")))
        self.add_node(Node("end", lambda ctx: print("End node executed")))

        # Define edges
        self.add_edge(Edge("start", "process"))
        self.add_edge(Edge("process", "end"))
```

### 6. Run the Graph

Create `src/run.py`:

```python
from graph import MyGraph

if __name__ == "__main__":
    graph = MyGraph()
    graph.run()
```

Execute:

```bash
python src/run.py
```

You should see:

```
Start node executed
Processing...
End node executed
```

### 7. Keep Dependencies Updated

Add the installed package to `requirements.txt`:

```bash
pip freeze > requirements.txt
```

Future collaborators can recreate the environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

You now have a clean environment, the LangGraph package installed, and a minimal project scaffold ready for further development!

## Building Your First Graph

Below is a minimal example that demonstrates how to set up a LangGraph for a simple question‑answering flow.  
We’ll define three nodes: **Ask**, **Search**, and **Answer**. The edges will route the user’s question to the search node and then to the answer node.

```python
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_retriever_node
from langchain_community.llms import OpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 1. Define the graph state
class QAState:
    question: str
    context: str = ""

# 2. Node: Ask (captures the user’s question)
def ask_node(state: QAState) -> QAState:
    # In a real app, this would come from user input
    state.question = "What is the capital of France?"
    return state

# 3. Node: Search (retrieves context)
def search_node(state: QAState) -> QAState:
    # Dummy retrieval – replace with a real vector store or API
    state.context = "Paris is the capital of France."
    return state

# 4. Node: Answer (generates the final answer)
def answer_node(state: QAState) -> QAState:
    llm = OpenAI()
    prompt = f"Answer the following question using the provided context:\n\nContext: {state.context}\n\nQuestion: {state.question}"
    response = llm(prompt)
    state.context = response  # Store answer for potential future use
    print(f"Answer: {response}")
    return state

# 5. Build the graph
graph = StateGraph(QAState)

# Add nodes
graph.add_node("ask", ask_node)
graph.add_node("search", search_node)
graph.add_node("answer", answer_node)

# Define edges
graph.add_edge(START, "ask")
graph.add_edge("ask", "search")
graph.add_edge("search", "answer")

# Compile the graph
qa_graph = graph.compile()

# Run the graph
qa_graph.invoke({})
```

**What happens?**

1. **Ask**: Sets the user’s question.  
2. **Search**: Looks up relevant context (here mocked).  
3. **Answer**: Feeds the question and context to an LLM and prints the answer.

You can extend this template by replacing the dummy retrieval with a real vector store, adding more nodes for clarification or summarization, and integrating it into a UI or API.

## Adding State and Memory

LangGraph lets you enrich a conversation by persisting state, context, and memory across turns. This makes the dialogue feel more natural and allows the system to remember user preferences, prior questions, or intermediate results.

### 1. Define a State Class

Create a dataclass (or plain dictionary) that holds the data you want to persist:

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class ConversationState:
    # Basic context
    user_name: str | None = None
    topic: str | None = None

    # History of messages (for context or debugging)
    messages: List[str] = field(default_factory=list)

    # Domain‑specific memory
    last_query: str | None = None
    last_result: str | None = None
```

### 2. Attach State to a Graph

When building the graph, pass the state object to the `StateGraph` constructor. All nodes can then read or mutate it.

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(ConversationState)

@graph.node
def greet(state: ConversationState):
    state.user_name = "Alice"          # Set a value
    state.messages.append("Hi!")       # Append to history
    return {"messages": state.messages}
```

### 3. Use Context in Nodes

Nodes can access the whole state, making decisions based on previous turns:

```python
@graph.node
def ask_topic(state: ConversationState):
    if not state.topic:
        return {"messages": ["What would you like to talk about?"]}
    # Continue with the chosen topic
    return {"messages": [f"Great, let's discuss {state.topic}."]} 
```

### 4. Persist Memory Across Sessions

If you want the conversation to remember information between separate runs (e.g., after the user leaves the chat), store the state in an external database or a simple file:

```python
import json

def save_state(state: ConversationState, path: str = "state.json"):
    with open(path, "w") as f:
        json.dump(state.__dict__, f)

def load_state(path: str = "state.json") -> ConversationState:
    with open(path, "r") as f:
        data = json.load(f)
    return ConversationState(**data)
```

Load the state before starting the graph and save it after each turn.

### 5. Leverage Built‑in Memory Nodes

LangGraph offers helper nodes for common memory patterns:

- **MemoryNode**: Stores arbitrary data in a key‑value store.
- **HistoryNode**: Keeps a rolling window of past messages.
- **ContextualPrompt**: Injects state variables into LLM prompts automatically.

Example:

```python
from langgraph.memory import MemoryNode

memory = MemoryNode()
graph.add_node("memory", memory)

@graph.node
def summarize(state: ConversationState, memory=memory):
    # Retrieve previous summary
    summary = memory.get("summary", "")
    # Update with new info
    memory.set("summary", f"{summary} {state.last_result}")
    return {"messages": [f"Updated summary: {memory.get('summary')}"]} 
```

### 6. Keep State Light and Fast

- Store only what you need; large blobs can slow down the graph.
- Use simple types (strings, ints, lists) that serialize quickly.
- If you need complex structures, serialize them to JSON or use a dedicated cache.

### 7. Test State Flow

Add unit tests that simulate a sequence of turns and assert that the state evolves as expected:

```python
def test_state_flow():
    state = ConversationState()
    state = greet(state)
    assert state.user_name == "Alice"
    state = ask_topic(state)
    assert "What would you like to talk about?" in state.messages[-1]
```

By carefully designing the state, injecting context into nodes, and persisting memory, you can create conversational agents that feel coherent, personalized, and engaging.

## Advanced Features and Customization

LangGraph is intentionally lightweight, but the real power comes from customizing how your graph behaves and how you observe it in production. Below are the most common advanced knobs you’ll want to tweak: custom handlers, structured logging, runtime monitoring, and hooks to external services.

---

### 1. Custom Handlers

Handlers let you intercept and modify messages as they flow through the graph. They are useful for:

- **Input validation** or transformation
- **Dynamic routing** based on context
- **Adding metadata** (e.g., user IDs, timestamps)

```python
from langgraph import Handler, Graph

class UppercaseHandler(Handler):
    def __call__(self, message, **kwargs):
        # Convert all text to uppercase before passing it on
        message["text"] = message["text"].upper()
        return message

graph = Graph()
graph.add_handler("pre_process", UppercaseHandler())
```

You can chain multiple handlers, and each handler receives the same `kwargs` dict, making it trivial to share state between them.

---

### 2. Structured Logging

Logging is essential for debugging complex flows. LangGraph ships with a simple logger interface that you can plug into your own logging framework.

```python
import logging
from langgraph import Graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langgraph")

class LoggingHandler:
    def __init__(self, logger):
        self.logger = logger

    def __call__(self, message, **kwargs):
        self.logger.info(
            f"Node {kwargs['node_name']} received: {message.get('text')}"
        )
        return message

graph = Graph()
graph.add_handler("post_process", LoggingHandler(logger))
```

Because the handler receives the node name via `kwargs['node_name']`, you can correlate logs with specific graph steps.

---

### 3. Monitoring & Metrics

For production workloads, expose metrics to Prometheus or any other monitoring stack. LangGraph can emit custom metrics through a `MetricsHandler`.

```python
from prometheus_client import Counter

request_counter = Counter("langgraph_requests_total", "Total number of requests")

class MetricsHandler:
    def __call__(self, message, **kwargs):
        request_counter.inc()
        return message

graph.add_handler("pre_process", MetricsHandler())
```

You can also hook into the graph’s event system to emit latency histograms or error counts.

---

### 4. Integrating External Services

LangGraph’s modular architecture makes it trivial to plug in third‑party services:

| Service | Typical Use‑Case | Example Hook |
|---------|------------------|--------------|
| **OpenAI / Azure** | LLM calls | `LLMNode` with `model="gpt-4o-mini"` |
| **Redis** | State persistence | `RedisStore` as the graph’s storage backend |
| **Slack / Teams** | Real‑time notifications | Custom `SlackHandler` that posts when a node fails |
| **Sentry** | Error tracking | `SentryHandler` that captures exceptions |

```python
from langgraph import Graph, LLMNode
from langgraph.integrations.redis import RedisStore
from langgraph.integrations.sentry import SentryHandler

# Persist state in Redis
store = RedisStore(url="redis://localhost:6379/0")
graph = Graph(store=store)

# Add LLM node
graph.add_node("chat", LLMNode(model="gpt-4o-mini"))

# Capture errors in Sentry
graph.add_handler("post_process", SentryHandler(dsn="your-dsn"))
```

You can also use *middleware* style wrappers around nodes to add retry logic, circuit breaking, or rate limiting.

---

### Quick Checklist for Production

- [ ] Add **validation handlers** to guard against malformed input.
- [ ] Enable **structured logging** for each node.
- [ ] Emit **metrics** for request counts, latencies, and error rates.
- [ ] Persist graph state in **Redis** or **PostgreSQL**.
- [ ] Hook into **Sentry** or **Datadog** for error monitoring.
- [ ] Add **retry logic** for external API calls.

With these customizations, your LangGraph application will be robust, observable, and ready for scale. Happy building!

## Common Pitfalls and Debugging Tips

- **Missing or Incorrect Agent Configuration**  
  *Problem:* Agents often fail because the `llm_config` or `tools` list is incomplete or mis‑typed.  
  *Debug:*  
  ```python
  print(agent.llm_config)   # Verify the LLM settings
  print(agent.tools)        # Ensure all tool names match the registry
  ```
  *Fix:* Double‑check the names and parameters against the official documentation.

- **Unresolved Tool Dependencies**  
  *Problem:* A tool raises `ModuleNotFoundError` or fails silently.  
  *Debug:*  
  ```bash
  pip list | grep <tool_name>
  ```
  *Fix:* Install the missing package or add the tool to the `ToolRegistry`.

- **Graph Cycle or Infinite Loop**  
  *Problem:* The graph never reaches a terminal node, causing a stack overflow or endless execution.  
  *Debug:*  
  ```python
  graph.visualize()  # Inspect the graph structure
  ```
  *Fix:* Add a proper `is_terminal` check or a maximum hop count.

- **State Mutation Errors**  
  *Problem:* The `state` object is mutated incorrectly, leading to unexpected results.  
  *Debug:*  
  ```python
  from copy import deepcopy
  old_state = deepcopy(state)
  # run step
  print("Diff:", {k: (old_state[k], state[k]) for k in state if old_state[k] != state[k]})
  ```
  *Fix:* Use immutable structures or ensure updates are performed via `state.update(...)`.

- **LLM Prompt Formatting Issues**  
  *Problem:* The prompt template contains syntax errors or placeholders that don’t resolve.  
  *Debug:*  
  ```python
  print(agent.build_prompt(state))  # See the exact prompt sent to the LLM
  ```
  *Fix:* Validate the Jinja2 template and escape any special characters.

- **Timeouts and Rate Limits**  
  *Problem:* The application stalls or throws a `TimeoutError` when calling an LLM or external API.  
  *Debug:*  
  ```python
  import logging
  logging.getLogger("langgraph").setLevel(logging.DEBUG)
  ```
  *Fix:* Increase `timeout` in the LLM config or implement exponential back‑off.

- **Incorrect Edge Conditions**  
  *Problem:* Conditional edges (`if`/`else`) never fire, causing dead ends.  
  *Debug:*  
  ```python
  print("Edges:", graph.edges)
  for edge in graph.edges:
      print(edge, edge.condition(state))
  ```
  *Fix:* Verify that the condition functions return booleans and reference the correct state keys.

- **State Serialization Issues**  
  *Problem:* When persisting state (e.g., to a database), non‑serializable objects cause failures.  
  *Debug:*  
  ```python
  import json
  try:
      json.dumps(state)
  except TypeError as e:
      print(e)
  ```
  *Fix:* Convert complex objects to primitives before serialization or use a custom encoder.

- **Missing `__repr__` for Custom States**  
  *Problem:* Debug prints are unreadable for custom state classes.  
  *Debug:*  
  ```python
  print(state)
  ```
  *Fix:* Implement `__repr__` or `__str__` for clearer output.

- **Version Mismatch**  
  *Problem:* Using an older LangGraph API with a newer LLM provider SDK leads to attribute errors.  
  *Debug:*  
  ```bash
  pip show langgraph
  pip show openai
  ```
  *Fix:* Update both packages to compatible versions (`pip install --upgrade langgraph openai`).

---

### Quick Debugging Checklist

| Step | What to Check | How to Verify |
|------|---------------|---------------|
| 1 | Agent config | `print(agent.llm_config)` |
| 2 | Tool registry | `print(agent.tools)` |
| 3 | Graph structure | `graph.visualize()` |
| 4 | Prompt output | `print(agent.build_prompt(state))` |
| 5 | State diffs | `deepcopy` comparison |
| 6 | Logs | Set logger to DEBUG |
| 7 | Dependencies | `pip list` |
| 8 | Version compatibility | `pip show` |

Following this guide should help you pinpoint and resolve most common LangGraph issues quickly. Happy graphing!

## Next Steps and Resources

- **Official Documentation**  
  * Deep dive into the LangGraph API, advanced patterns, and best‑practice guides on the [LangGraph docs site](https://docs.langgraph.org/).  
- **Starter Tutorials**  
  * Follow the “Building a Conversational Agent” walkthrough in the docs or the [LangGraph YouTube playlist](https://youtube.com/playlist?list=PLLangGraph).  
- **Community Forums & Chat**  
  * Join the LangGraph Discord server for real‑time help and idea exchange.  
  * Participate in the #langgraph channel on the LangChain Slack workspace.  
- **GitHub Repository**  
  * Browse open issues, contribute fixes, or submit pull requests to the official [LangGraph GitHub repo](https://github.com/langchain-ai/langgraph).  
- **Build Your Own Projects**  
  * Create a simple FAQ bot, a data‑driven recommendation engine, or a multi‑step workflow that incorporates external APIs.  
  * Experiment with custom connectors, state schemas, and graph traversal logic.  
- **Advanced Topics**  
  * Learn about dynamic node creation, graph‑based memory management, and integrating LangGraph with other frameworks (e.g., LangChain, LlamaIndex).  
  * Study performance tuning: batching, caching, and parallel execution.  
- **Courses & Workshops**  
  * Enroll in the “AI Agent Programming with LangGraph” course on Coursera or the workshop offered at the annual AI Summit.  
- **Books & Papers**  
  * “Building Intelligent Agents with LangGraph” (upcoming book).  
  * Read research papers on graph‑based agent architectures for deeper theoretical grounding.  

Use these resources to transition from foundational knowledge to building sophisticated, production‑ready LangGraph applications.
