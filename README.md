# LangGraph — Revision Notes

Study repo. One folder per topic, each with Jupyter notebooks, all sharing a single venv at
`venv/` and one `.env` at the root. Open a notebook in VS Code with the `venv` kernel, or
double-click the folder's `run.bat` to start Jupyter with the venv. The code lives in the
notebooks (and the `.py` files for the Streamlit app); this file is only the learnings.

| Folder | Topic |
|---|---|
| `LG_SequentialWorkflow` | State, nodes, edges, compile, invoke (BMI calculator, LLM Q&A, prompt chaining) |
| `LG_ParallelWorkflow` | Fan out / fan in, partial state updates, reducers (batsman stats, LLM essay evaluation) |
| `LG_ConditionalWorkflow` | Conditional edges, router functions (quadratic solver, LLM review reply) |
| `LG_IterativeWorkflow` | Loops, stop conditions, recursion limit (number guessing, LLM tweet improver) |
| `LG_Persistence` | Message state, `add_messages`, chat loop, persistence (checkpointer, threads) |
| `LG_Chatbot` | Streamlit chat UI over the LangGraph backend, streaming replies, multiple threads, switching conversations, SQLite persistence, LangSmith tracing, tools (search, calculator, stocks) |
| `LG_Tools` | Tool calling: `@tool`, `bind_tools`, `ToolNode`, `tools_condition` |

---

## 1. Sequential Workflow

Notebooks: `bmi_workflow.ipynb`, `llm_workflow.ipynb`, `prompt_chaining.ipynb`

**The mental model**
- A LangGraph app is a **graph**: **nodes** do the work, **edges** decide what runs next,
  and one shared **state** flows through every node.
- **Sequential** = nodes in a straight line, each runs once, in order.
  `START -> calculate_bmi -> label_bmi -> END`

**State**
- Defined as a `TypedDict`. It's the graph's single source of data. Nodes don't pass
  values to each other directly; they write to state and the next node reads from it.
- You only pass the fields you have at `invoke` time; nodes fill in the rest.
- `invoke` returns the **final state**, so every intermediate value is still there to inspect.

**Nodes**
- A node is just a plain Python function: takes the state, returns the updated state.
- The graph doesn't care what happens inside. Maths (BMI) and an LLM call look the same
  to LangGraph.

**Edges and the lifecycle**
- `START` and `END` are built-in markers for where the graph begins and stops.
- Always three steps: **build** the graph (add nodes and edges), then **compile** it
  (checks the structure and makes it runnable), then **invoke** it with the initial state.
- `workflow.get_graph().draw_mermaid_png()` draws the graph, which helps check the wiring.

**Prompt chaining**
- Several LLM calls in a row, where one call's output is the next call's input
  (title -> outline -> blog). The state is what carries the result between steps.
- Small focused prompts beat one giant prompt, and each step's output can be checked
  on its own.

**Interview questions**
- *What are the three building blocks of a LangGraph graph?* State (shared data, a
  `TypedDict`), nodes (functions that read and update state), edges (what runs next).
- *Why do you have to call `compile()`?* It checks the structure (e.g. no unreachable nodes,
  edges point to real nodes) and turns the builder into a runnable you can `invoke`/`stream`.
- *What does `invoke` return?* The final state, including every intermediate value the
  nodes wrote, not just the last node's output.
- *Why use prompt chaining instead of one big prompt?* Each step is simpler, so results
  are better, and you can inspect, test or retry one step on its own.

---

## 2. Parallel Workflow

Notebooks: `simple_parallel.ipynb`, `llm_parallel.ipynb`

**The mental model**
- Nodes that don't depend on each other run **at the same time**, then their results
  are combined. Batsman stats: strike rate, balls per boundary and boundary % are
  independent, so they run in parallel and a `summary` node joins them.
- **Fan out:** add several edges from the same node (here `START`) to the parallel nodes.
- **Fan in:** add an edge from each parallel node into one node. LangGraph makes that
  node **wait until all of them have finished** before running it.

**Return only what you change (the big gotcha)**
- In parallel, a node must return a **partial dict** of just the keys it updates,
  e.g. `{'sr': 200.0}`, not the whole state.
- Returning the full state means every parallel node writes every key in the same step,
  and LangGraph refuses:
  `InvalidUpdateError: At key 'a': Can receive only one value per step.`
- LangGraph merges the partial dicts into the state for you. Returning partial dicts works
  in sequential graphs too, so it's the safer habit everywhere.

**When parallel nodes must write the same key: reducers**
- Essay evaluation: three LLM judges (language, analysis, clarity) each produce a score,
  and all three want to put it in `scores`. Normally that's the same error as above.
- Fix: give the key a **reducer** with `Annotated`. The reducer says how to combine
  values instead of overwriting:
  `scores: Annotated[list[int], operator.add]`
- Each node returns `{'scores': [7]}` (a one-item list), and `operator.add` joins the
  lists, so the fan-in node sees `[5, 8, 8]`.
- Keys without a reducer get **overwritten**; keys with one get **combined**.

**LLM judges with structured output**
- `model.with_structured_output(PydanticModel)` makes the LLM return an object
  (`feedback`, `score`) instead of free text, so a node can read `result.score` directly.
- Parallel LLM calls also save time: the three judges run together, not one after another.

**Interview questions**
- *How do you run nodes in parallel in LangGraph?* Fan out: add edges from one node to
  several nodes. Fan in: add an edge from each of them into one node, which waits for all.
- *Why must parallel nodes return partial dicts?* They run in the same step; if each returns
  the whole state, several nodes write the same keys at once and LangGraph raises
  `InvalidUpdateError`.
- *What is a reducer and when do you need one?* A function that says how to combine updates
  to a key (`Annotated[list, operator.add]`), needed when several nodes write the same key.
- *Without a reducer, what happens to a key that two nodes update in sequence?* The later
  value overwrites the earlier one.

---

## 3. Conditional Workflow

Notebooks: `simple_conditional.ipynb`, `llm_conditional.ipynb`

**The mental model**
- The graph **picks one path** based on the state, like an `if/elif/else`. Quadratic solver:
  the discriminant decides between `real_roots`, `repeated_roots` and `no_real_roots`,
  and only that one branch runs.
- Parallel = all branches run. Conditional = exactly one branch runs.

**The router function**
- `add_conditional_edges('calculate_discriminant', check_condition)` replaces a normal edge.
  After that node, LangGraph calls `check_condition` to decide where to go next.
- The router is **not a node**: it doesn't change the state. It only reads it and
  returns the **name** of the next node as a string.
- Type the return as `Literal['real_roots', 'repeated_roots', 'no_real_roots']`. That's
  how LangGraph knows every possible destination, so the graph compiles and draws the
  branches correctly.
- Each branch still needs its own edge to `END` (or to wherever the paths join back).

**Letting an LLM make the decision**
- Customer review reply: an LLM classifies the review as positive or negative, and the
  router sends it down the right branch. Positive gets a thank-you; negative goes to
  `run_diagnosis` (issue type, tone, urgency) and then a tailored support reply.
- The LLM node does the thinking and writes a label into state. The router stays plain
  Python that reads that label. Keep routers dumb and fast.
- Use **structured output with a `Literal`** (`Literal['positive', 'negative']`) for the
  label. Free text like "The sentiment is mostly positive." would break the `if` check.
- A branch can be several nodes long (`run_diagnosis -> negative_response`); it's just
  normal edges after the split.

---

## 4. Iterative Workflow

Notebooks: `simple_iterative.ipynb`, `llm_iterative.ipynb`

**The mental model**
- A **loop**: a conditional edge that can point **back** to an earlier node, so the graph
  repeats steps until a condition is met. Number guessing: `make_guess -> check_guess`,
  then back to `make_guess` until the hint is `correct`.
- It's the same `add_conditional_edges` as before. The only difference is that one of the
  router's destinations is a node that already ran.
- The router returns `END` to stop. In the `Literal` type hint, `END` is written as
  `'__end__'`.

**State carries progress between rounds**
- Each round reads what the previous round left behind (`low`, `high`, `attempts`) and
  updates it. Without state, a loop would just repeat the same step forever.
- Keep a counter like `attempts` in state; it's how you see or cap the number of rounds.

**Always have a way out**
- If the stop condition never comes true, LangGraph stops the run with
  `GraphRecursionError: Recursion limit of ... reached`. It's a safety net, not a stop
  condition.
- Change it per run with `workflow.invoke(state, {'recursion_limit': 50})`. Better: put a
  max-attempts check in the router yourself.

**LLM loop: generate → evaluate → optimize**
- Tweet writer: one LLM writes, a second LLM judges (`approved` / `needs_improvement`
  + feedback), and if it's not good enough an optimizer rewrites using that feedback,
  then the judge checks again. This **generator–evaluator–optimizer** loop is how you
  get an LLM to improve its own output.
- The router has **two exits**: approved, *or* `iteration >= max_iteration`. A strict LLM
  judge may never approve, so the counter is what guarantees the loop ends.
- Use a **different (bigger) model as the judge**. The same model tends to approve its
  own work.
- Keep a history with a reducer (`tweet_history: Annotated[list[str], operator.add]`).
  Each round returns `[new_tweet]` and it's appended, so the final state shows every
  version, not just the last.

**gpt-oss gotcha: empty replies**
- gpt-oss models "think" before answering. Sometimes all the output goes into that
  hidden reasoning and `.content` comes back **empty**, which then poisons the next
  round of the loop.
- Fix: `ChatGroq(model='openai/gpt-oss-20b', reasoning_effort='low')` for simple
  writing tasks.

---

## 5. Persistence

Notebooks: `simple_chatbot.ipynb`, `persistence.ipynb`

**The mental model**
- A chatbot is the smallest graph possible, `START -> chat_node -> END`. The state is just
  a **list of messages**: the node sends the whole list to the LLM and adds the reply.
- The chat loop (`while True: input() ...`) lives **outside** the graph. Each message you
  type is one `invoke`.

**`add_messages`: the reducer for chat**
- `messages: Annotated[list[BaseMessage], add_messages]`. Like `operator.add`, it
  **appends** new messages instead of overwriting the list, so the node only returns
  `{'messages': [response]}`.
- It's smarter than `operator.add`: it also handles message IDs (a message with the same
  ID replaces the old one instead of being duplicated).

**Why the simple bot has no memory**
- Every `invoke` starts from a **fresh state**. Tell it "my name is Priyansh", then ask
  "what is my name?" in the next invoke, and it doesn't know.
- The state only lives for one run of the graph. Remembering across turns needs the
  state to be **saved between invokes**. That's persistence.

**Persistence: a checkpointer gives the graph memory**
- `graph.compile(checkpointer=InMemorySaver())`. The graph code doesn't change at all;
  only the compile line does.
- The checkpointer **saves the state after every step**. On the next invoke it loads the
  saved state first, then `add_messages` appends the new message to the old ones.
  So the bot remembers "my name is Priyansh".
- **`thread_id` = one conversation.** Pass it on every call:
  `config = {'configurable': {'thread_id': '1'}}`. Same thread = continues;
  new thread = starts empty. That's how one bot serves many users or chats separately.
- `chatbot.get_state(config)` reads the saved state of a thread (the full chat so far).
- `chatbot.get_state_history(config)` lists **every checkpoint**, newest first, one per
  step, so each turn leaves several snapshots behind. That history is what later makes
  things like resuming after a crash and going back in time possible.
- `InMemorySaver` keeps everything in RAM, so it's gone when the kernel restarts. For
  real apps you swap in a database-backed checkpointer (e.g. SQLite); the rest stays the same.

---

## 6. Chatbot (Streamlit UI)

Files: `chatbot_backend.py` (the graph), `streamlit_frontend.py` (the UI). Start with `run.bat`.

**Backend / frontend split**
- The backend is the persistence chatbot from section 5, moved into a `.py` file that
  exposes one object: `chatbot` (the compiled graph with a checkpointer).
- The frontend only does UI: show messages, take input, call `chatbot.invoke(...)`.
  It knows nothing about nodes or edges, so either side can change without touching the other.

**How Streamlit works (and why `session_state`)**
- Streamlit **reruns the whole script from the top** every time you send a message.
  Normal variables are wiped on each rerun.
- `st.session_state` survives reruns, so the list of messages shown on screen lives there.
  Without it the chat window would empty after every message.
- Two separate memories: `session_state` is what the **UI displays**; the checkpointer
  (by `thread_id`) is what the **LLM remembers**. They hold the same chat for different jobs.

**Chat building blocks**
- `st.chat_input('Type here')` is the input box at the bottom; it returns the text once sent.
- `with st.chat_message('user' / 'assistant'):` draws a chat bubble with the right avatar.

**Streaming: show the reply as it's typed**
- `chatbot.invoke` waits for the whole reply. `chatbot.stream(..., stream_mode='messages')`
  instead yields the LLM's reply **token by token** as `(message_chunk, metadata)` pairs,
  while the graph is still running.
- `st.write_stream(generator)` draws each piece as it arrives and **returns the full text**
  at the end, which is what gets saved into `session_state`.
- Only the frontend changed. The graph and checkpointer are the same, and the streamed
  reply is still saved to memory when the run finishes.
- With gpt-oss, many chunks have **empty `.content`** (that's the hidden reasoning).
  `st.write_stream` just skips them.
- `st.write_stream` renders Markdown, so show past messages with `st.markdown` too,
  otherwise replies look different after the page reruns.

**Sidebar and multiple conversations (threads)**
- Each chat gets its own **`thread_id`**, made with `uuid.uuid4()` so it's unique, and
  kept in `session_state` so it survives reruns. The config passed to the graph is built
  from it, so the checkpointer stores each chat separately.
- **Start Chat** = new thread id + empty the on-screen `message_history`. The old chat is
  still saved in the checkpointer under its own id. The bot only "forgets" because the
  next message goes to a different thread.
- All thread ids are kept in a list in `session_state` (`chat_threads`) and listed in the
  sidebar under "My Conversations", newest first, as **clickable buttons**.

**Loading an old conversation**
- Clicking a thread's button switches `thread_id` to it and refills `message_history`
  from the checkpointer: `chatbot.get_state(config).values['messages']`.
- The checkpointer stores LangChain message objects; the UI wants `{'role', 'content'}`
  dicts. Convert: `HumanMessage` → `'user'`, anything else → `'assistant'`.
- A thread with no messages yet has **empty `values`**, so use `.get('messages', [])`.
- After switching, the chat continues where it left off, because new messages go to that
  thread's saved state. Threads stay isolated: facts told in one chat aren't known in another.
- **Readable names instead of ids:** a `thread_names` dict in `session_state` maps
  `thread_id -> name`; a chat with no messages shows "New Chat". The id stays the real
  identity; the name is only a label.
- **LLM-written titles:** after the first message, `generate_chat_title()` in the backend
  asks the LLM for a max-5-word title ("Jaipur Budget 3-Day Plan"). It's a plain
  `model.invoke`, **outside the graph**, so the title never enters the chat history or the
  checkpointer. Done once per chat, so it costs one extra LLM call per conversation.
  (Simpler alternative: cut the first message to 30 characters, no LLM call.)
- The sidebar is drawn **before** the new message is handled, so it still shows the old
  label. Call `st.rerun()` once after naming so the new name appears straight away.
- Give buttons made in a loop a **unique `key`** (`key=thread_id`). Two buttons with the
  same label and no key make Streamlit raise a duplicate-element error.
- `st.sidebar.title / .button / .header / .text` put elements in the sidebar instead
  of the main page.
- After the button changes the thread, call `st.rerun()` so the rest of the page is drawn
  with the new thread straight away.

**Database persistence: SQLite checkpointer**
- `InMemorySaver` loses every chat when the server stops. Swap it for
  `SqliteSaver(conn=sqlite3.connect('chatbot.db', check_same_thread=False))` (package
  `langgraph-checkpoint-sqlite`) and chats are saved in a file instead. Nothing else in
  the graph changes, which is the whole point of the checkpointer design.
- `check_same_thread=False` because Streamlit handles reruns on different threads, all
  using the one connection made at import.
- **Listing old chats:** `checkpointer.list(None)` returns every checkpoint in the
  database, newest first. Collect the unique `thread_id`s and that's the conversation list
  for the sidebar on startup.
- **Anything not in the graph state is lost on restart.** Chat titles were only in
  `session_state`, so they're now a `title` key in `ChatState`, written with
  `chatbot.update_state(config, {'title': ...})` and read back with `get_state`.
  `update_state` writes into a thread's saved state **without running the graph**.
- **What's inside `chatbot.db`:** two tables. `checkpoints` has one row per saved
  snapshot (thread_id, checkpoint_id, parent, the state stored as binary msgpack), and
  `writes` has the individual node outputs. A short chat already makes many rows, since
  every step is saved. Open it with any SQLite viewer to look around.
- The `.db` file holds your chats, so it's in `.gitignore`.

**Observability: LangSmith tracing, grouped by thread**
- Tracing needs **no code**: `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` +
  `LANGSMITH_PROJECT` in `.env`, and `load_dotenv()` makes every LangChain/LangGraph call
  send a trace (graph → node → LLM call, with inputs, outputs, tokens and latency).
- **One trace per `invoke`/`stream`**, so a conversation is many separate traces. To see
  them as one chat, put the id in **`metadata`**: `{'metadata': {'thread_id': ...}}`.
  LangSmith's **Threads** tab groups traces by a `thread_id` (or `session_id` /
  `conversation_id`) metadata key.
- Same id, two jobs, two places in the config: `configurable.thread_id` tells the
  **checkpointer** which chat to load; `metadata.thread_id` tells **LangSmith** how to group.
- `run_name` names the trace (`chat_turn` instead of the default `LangGraph`), which makes
  the run list readable. Any `model.invoke(..., config={'run_name': ...})` works too.
- The title call (`generate_chat_title`) gets a run name but **no thread_id**, on purpose:
  it isn't a chat turn, so it shouldn't show up inside the conversation's thread.

**Tools in the chatbot** (the section 7 pattern, inside the real app)
- Three tools: `DuckDuckGoSearchRun()` (a ready-made LangChain tool, no API key),
  a `calculator(first_num, second_num, operation)` and `get_stock_price(symbol)` using
  `yfinance` (real prices, no API key). The graph becomes
  `chat_node -> tools -> chat_node` with `ToolNode` + `tools_condition`.
- **Streaming gets noisier:** `stream_mode='messages'` now also yields `ToolMessage`s
  (raw tool output) and AI chunks that only request a tool. Yield **only `AIMessage`
  content** to `st.write_stream`, or the raw search results get printed as the reply.
- Use the `ToolMessage`s as a signal instead: show an `st.status('Using tool…')` box and
  mark it complete at the end, so the user knows why the reply is taking longer.
- **Loading old chats** must skip `ToolMessage`s and AI messages with empty content
  (tool requests), or they show up as blank or raw bubbles.
- One docstring line per tool decides when the LLM picks it ("ticker symbol, e.g. AAPL,
  or INFY.NS for NSE India"). It's the tool's instruction manual.

**Going async (needed for MCP tools)**
- MCP tools are **async-only**, so the graph must run async: `async def chat_node` with
  `await model.ainvoke(...)`, `chatbot.astream(...)`, `aget_state`, `aupdate_state`, and
  `AsyncSqliteSaver` (with an `aiosqlite` connection) instead of `SqliteSaver`.
- Streamlit is **not async**. Fix: start **one asyncio event loop in a background thread**
  and send every coroutine to it with `asyncio.run_coroutine_threadsafe(coro, loop).result()`.
  One loop for everything, because the async DB connection is tied to the loop it was made on.
- **Streaming across the gap:** the async stream runs on that loop and puts each chunk in a
  `queue.Queue`; a normal generator on the Streamlit side reads the queue until a `None`
  "done" marker. So `st.write_stream` still gets a plain generator.
- The backend now hides LangGraph from the frontend behind small helpers (`stream_reply`,
  `get_thread_values`, `save_title`), so all the async plumbing lives in one file.
- Sync tools still work in an async graph: `ToolNode` runs them in a thread.

**MCP: tools from a separate server**
- **MCP (Model Context Protocol)** is a standard way for an app to get tools from a separate
  program, the **MCP server**. Write a tool once and any MCP client (this chatbot, Claude
  Desktop, an IDE) can use it.
- Server side (`calculator_mcp_server.py`): `FastMCP('calculator')`, decorate functions with
  `@mcp.tool()`, then `mcp.run(transport='stdio')`. The docstring is still what the LLM reads.
- Client side: `MultiServerMCPClient({...})` from `langchain-mcp-adapters` starts the server
  as a subprocess (`'transport': 'stdio'`, `'command': sys.executable`, `'args': [path]`) and
  `await client.get_tools()` returns normal LangChain tools to mix with local ones.
- `get_tools()` and the MCP tools are **async-only**, which is why the chatbot went async first.
- The graph doesn't change at all: MCP tools go into the same `tools` list, `bind_tools` and
  `ToolNode`. Only *where the tool runs* changed.
- **How to tell the MCP server really ran:** its `ToolMessage.content` is a list of content
  blocks (`[{'type': 'text', 'text': '31792582.26'}]`), not the plain string a local tool returns.
- The LLM still decides whether to call it: "987 ÷ 21" it did in its head, so to test the MCP
  path, ask it explicitly to "use the calculator tool".

---

## 7. Tools

Notebooks: `tool_calling.ipynb`

**The mental model**
- A **tool** is a normal Python function the LLM can ask to use: maths, a search, an API.
  The LLM **never runs it**. It replies with a *tool call* (name + arguments), your code
  runs the function, and the result goes back to the LLM to write the final answer.
- In LangGraph that's a loop: `chat_node -> tools -> chat_node -> ... -> END`, repeating
  until the LLM answers without asking for a tool.

**Making and binding tools**
- `@tool` on a function makes it a tool. The LLM only sees the **name, docstring and
  argument types**, so the docstring is what tells it *when* to use the tool. Write it well.
- `model.bind_tools(tools)` tells the LLM which tools exist. Calling it directly shows the
  raw idea: `content` is empty and `tool_calls` holds `[{'name': 'multiply', 'args': {...}}]`.

**The two prebuilt pieces**
- `ToolNode(tools)`: a ready-made node that reads the tool calls in the last AI message,
  runs the functions and adds the results as `ToolMessage`s.
- `tools_condition`: a ready-made router. Tool calls in the last message → go to the node
  named **`'tools'`**; none → `END`. (Name the node `'tools'` or it won't find it.)
- `add_edge('tools', 'chat_node')` closes the loop so the LLM sees the tool's result.

**What the message list shows**
- `human -> ai (tool call) -> tool (result) -> ai (answer)`. Every step is a message in
  state, which is exactly why the `add_messages` reducer from section 5 is needed.

**Gotchas**
- The LLM **decides** whether to use a tool. "Hi" uses none; easy steps it may do itself
  (it called `multiply` for 23×47 but added the 12 on its own). For must-be-exact work,
  say so in the prompt or docstring.
- Tool results come back as **strings**, whatever the function returned.
