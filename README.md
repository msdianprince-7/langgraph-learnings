# LangGraph — Revision Notes

Study repo. One folder per topic, each with Jupyter notebooks, all sharing a single venv at
`venv/` and one `.env` at the root. Open a notebook in VS Code with the `venv` kernel, or
double-click the folder's `run.bat` to start Jupyter with the venv. The code lives in the
notebooks; this file is only the learnings.

| Folder | Topic |
|---|---|
| `LG_SequentialWorkflow` | State, nodes, edges, compile, invoke (BMI calculator, LLM Q&A, prompt chaining) |
| `LG_ParallelWorkflow` | Fan out / fan in, partial state updates, reducers (batsman stats, LLM essay evaluation) |
| `LG_ConditionalWorkflow` | Conditional edges, router functions (quadratic solver, LLM review reply) |
| `LG_IterativeWorkflow` | Loops, stop conditions, recursion limit (number guessing, LLM tweet improver) |

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
