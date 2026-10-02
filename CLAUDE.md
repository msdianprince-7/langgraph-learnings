# LangGraph study repo: working rules

1. **Structure:** one folder per topic, named `LG_<Topic>` (e.g. `LG_Basics`, `LG_State`,
   `LG_ConditionalEdges`, `LG_Loops`, `LG_ToolCalling`, `LG_Memory`, `LG_HumanInTheLoop`,
   `LG_MultiAgent`). Go one topic at a time, in the order the user asks.
2. **Setup:** one shared venv at `D:\LangGraph\venv` and one `.env` at the root
   (holds `GROQ_API_KEY`, copied from `D:\LangChain\.env`). Never commit `.env` or `venv`.
   Each topic folder gets a `run.bat` that calls `..\venv\Scripts\python.exe`.
   Code is written as Jupyter notebooks (`.ipynb`); `run.bat` opens Jupyter with the venv.
   Use Groq models that work on this account: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`.
3. **Code style:** the user is learning, so write the smallest code that shows the concept.
   No error handling, caching, docstrings or helper abstractions unless asked.
   When asked to run a file, just show the output.
4. **Revision notes:** the root `README.md` is the user's revision notes, styled like
   `D:\LangChain\README.md`: a table of folders and topics at the top, then one numbered
   section per topic. Each section holds the **learnings** (concepts, why it works, gotchas)
   as short bullets, not a copy of the notebook code; the code lives in the notebooks.
   At most a one-line snippet when a specific call is worth remembering. Update it after
   each topic.
5. **Git:** one commit per finished topic, pushed to `main`.
