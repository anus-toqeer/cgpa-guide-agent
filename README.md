# CGPA Guider: MCP-Powered Agent

A terminal chatbot for PUCIT students that answers GPA/CGPA questions. The LLM does no math: it only decides which tool to call, and every number comes from Python functions served over **MCP**.

## Features
- **MCP client**: tools are fetched from a remote MCP server (`MCPAdapter`), not stored in the agent.
- **Persistent memory**: `AsyncSqliteSaver` saves each chat in `chats.db`, one thread per student name.
- **Streaming**: `astream` with `updates` and `messages` shows tool calls and a token-by-token answer.
- **Server-down handling**: tool calls that never got a result are repaired, so a chat is never stuck.

## Tools (served by the MCP server)
`get_semester_courses`, `marks_to_grade_points`, `calculate_semester_gpa`, `calculate_new_cgpa`, `required_gpa_for_target`, `get_remaining_credit_hours`, `save_report`

## Links
- MCP server code: https://github.com/anus-toqeer/my-mcp-server
- Hosted server (Prefect Horizon, private, needs an API key): `https://cgpa-calc.fastmcp.app/mcp`

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env     # then fill in your keys
python agent.py
```

`.env` values:
```
GROQ_API_KEY=...
MCP_URL=https://YOUR-SERVER-NAME.fastmcp.app/mcp
FASTMCP_KEY=...          # leave out if the server is public
```

To run against a local server instead: `cd server && python server.py`, then set `MCP_URL=http://127.0.0.1:8000/mcp` and remove `FASTMCP_KEY`.

## Commands
| Command | Action |
|---|---|
| `/switch` | change person (thread) |
| `/history` | show the saved chat, including tool calls |
| `/forget` | delete this person's chat |
| `quit` | exit |

## Tech
Python 3.13, LangChain `create_agent`, LangGraph checkpointer (SQLite), FastMCP, Groq `openai/gpt-oss-20b`.
