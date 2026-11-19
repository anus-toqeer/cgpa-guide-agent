import asyncio
import os
from fastmcp import Client
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from llm import llm

load_dotenv()
MCP_URL = os.getenv("MCP_URL")
KEY = os.getenv("FASTMCP_KEY")

SYSTEM_PROMPT = """You are a GPA/CGPA assistant for PUCIT BS(CS) students.
When a student describes their academic progress, be careful about semester
numbering:
- "I'm in semester 5" / "I'm a 5th semester student" means they are CURRENTLY
  STUDYING semester 5 and have COMPLETED semesters 1-4. Semester 5 itself is
  NOT yet finished and should be included in "remaining" hours.
- "I've completed semester 5" / "I just finished semester 5" means semester 5
  IS done, and only semesters 6-8 remain.
- If this is unclear from what the student said, ASK explicitly which they
  mean rather than assuming - this directly affects credit hour totals and
  the tools' current_semester parameter (which expects the LAST COMPLETED
  semester number).

When calling any tool, use the EXACT parameter names as defined in that
tool's function signature - do not paraphrase or shorten them (for example,
required_gpa_for_target's target parameter is named target_cgpa, not
target_gpa).

When reporting a required GPA, always state explicitly WHICH semesters that
GPA applies to (e.g. "across semesters 5 through 8 combined" or "in semester
5 alone") - never say just "this semester" when the number actually spans
multiple semesters.

CORE RULES:
1. You do NO arithmetic yourself. Every single number in your answers must come
   from a tool call. If you find yourself about to compute or estimate a number
   directly, stop and call the appropriate tool instead.

2. NEVER invent or guess marks, credit hours, a CGPA, or a semester number.
   If you need a piece of information to call a tool and the student hasn't
   given it to you, ask for it. Do not assume typical/average values.

3. Ask for missing information ONE OR TWO items at a time, in a natural,
   conversational way. Never dump a long form of 5+ questions on the student
   at once.

4. Math Deficiency courses (MD-001, MD-002) are non-credit, pass/fail, and are
   EXCLUDED from all GPA calculations. Ignore them entirely - do not ask about
   them, do not include them in credit hour totals. Every other course in the
   scheme of studies counts, including the 0.5-credit-hour Quran Translation
   courses each semester.

5. A required GPA above 4.0 is impossible in that specific window. When this
   happens, do NOT ask the student what to assume about other semesters.
   Instead, silently retry using a wider window, following this exact procedure:
   a. Call required_gpa_for_target using only the immediate next semester's
      credit hours as remaining_credit_hours.
   b. If the result is above 4.0, call get_remaining_credit_hours again for
      the next 2 semesters combined, and retry required_gpa_for_target with
      that larger number.
   c. Keep adding one more semester's worth of credit hours and retrying
      until the required GPA is at or below 4.0, or you reach semester 8
      (the maximum possible window).
   d. Report the SMALLEST window that works, phrased as advice: "Semester 5
      alone isn't enough, but across semesters 5-6 combined you'd need
      roughly X GPA, which is achievable."
   e. If even the full remaining curriculum (up to semester 8) requires
      above 4.0, then and only then tell the student the goal is not
      achievable under any semester combination.

6. Only offer to save a report (using save_report) after you've produced a
   real, complete result worth keeping - a computed semester GPA, a projected
   CGPA, or a target-GPA plan. Never offer to save after only asking a
   clarifying question, and never save anything without the student
   explicitly agreeing first.

You are stateless between tool calls - always rely on the conversation history
for context the student already gave you, and don't ask for information twice.
"""

async def list_threads(memory):
    """Names of everyone who has a saved chat."""
    names = set()
    async for cp in memory.alist(None):          # every saved checkpoint
        names.add(cp.config["configurable"]["thread_id"])
    return sorted(names)


async def show_history(app, config):
    """Print the saved chat of the current thread."""
    state = await app.aget_state(config)         # latest snapshot
    messages = state.values.get("messages", [])
    if not messages:
        print("(no saved messages)")
        return
    for m in messages:
        if m.type == "human":
            print("You:", m.content)
        elif m.type == "ai":
            if m.tool_calls:
                for call in m.tool_calls:
                    print(f"  [model asks for] {call['name']}({call['args']})")
            if m.content:
                print("Agent:", m.content)
        elif m.type == "tool":
            print(f"  [tool {m.name} returned] {m.text}")


def ask_name():
    return input("Who is chatting? (your name): ").strip().lower()

async def stream_reply(app, user, config):
    inputs = {"messages": [HumanMessage(user)]}
    print("Agent: ", end="", flush=True)

    async for chunk in app.astream(inputs, config,
                                   stream_mode=["updates", "messages"],
                                   version="v2"):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            # typing effect: only the model's own text, skip tool results
            if metadata["langgraph_node"] == "model" and token.text:
                print(token.text, end="", flush=True)

        else:  # "updates": a node just finished
            for node, update in chunk["data"].items():
                for msg in update["messages"]:
                    if node == "model" and msg.tool_calls:
                        for call in msg.tool_calls:
                            print(f"\n  [model asks for] {call['name']}({call['args']})")
                    elif node == "tools":
                        print(f"  [tool {msg.name} returned] {msg.text}")
    print()

async def repair_dangling(app, config):
    """Answer every tool call that never got a result, so the chat stays valid."""
    state = await app.aget_state(config)
    messages = state.values.get("messages", [])
    answered = {m.tool_call_id for m in messages if m.type == "tool"}

    fixes = []
    for m in messages:
        if m.type == "ai" and m.tool_calls:
            for call in m.tool_calls:
                if call["id"] not in answered:
                    fixes.append(ToolMessage(
                        content=("Error: this tool call failed because the tool server "
                                 "was unreachable at that moment. It does not affect "
                                 "later requests."),
                        tool_call_id=call["id"],
                        name=call["name"]))
    if fixes:
        # close the turn so the model doesn't treat the error as an open instruction
        fixes.append(AIMessage("Sorry, the tool server was unreachable, so I could not "
                               "finish that request. Please ask again and I will retry."))
        await app.aupdate_state(config, {"messages": fixes}, as_node="tools")
        return True
    return False

async def main():
      server = Client(MCP_URL, auth=KEY) if KEY else MCP_URL
      async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()

        async with AsyncSqliteSaver.from_conn_string("chats.db") as memory:
            app = create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT,
                               checkpointer=memory)

            print("Saved chats:", await list_threads(memory))
            name = ask_name()
            config = {"configurable": {"thread_id": name}}
            print("Commands: /switch  /history  /forget  quit\n")

            while True:
                user = input(f"{name}: ").strip()
                if not user:
                    continue
                if user.lower() in ("quit", "exit"):
                    break
                if user == "/switch":
                    name = ask_name()
                    config = {"configurable": {"thread_id": name}}
                    continue
                if user == "/history":
                    await show_history(app, config)
                    continue
                if user == "/forget":
                    await memory.adelete_thread(name)
                    print(f"Chat of '{name}' deleted.")
                    continue

                await repair_dangling(app, config)      # heal old stuck threads
                try:
                    await stream_reply(app, user, config)
                except Exception as e:
                    print(f"\n[!] Tool server problem: {type(e).__name__}")
                    if await repair_dangling(app, config):
                        print("    The failed tool call was marked as an error in your chat.")


asyncio.run(main())