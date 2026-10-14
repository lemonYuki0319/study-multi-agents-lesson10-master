import uuid
from langgraph.types import Command

from supervisor import create_supervisor


def print_separator():
    print("=" * 60)


def _print_stream_chunk(chunk):
    """Print tool calls and messages from a single stream chunk."""
    if not isinstance(chunk, dict):
        return
    for _, update in chunk.items():
        if not isinstance(update, dict):
            continue
        msgs = update.get("messages", [])
        for msg in msgs:
            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            if role == "tool":
                tool_name = getattr(msg, "name", "")
                print(f"  📎 [{tool_name} result]: {str(msg.content)[:200]}...")
            elif role == "ai":
                tool_calls = getattr(msg, "tool_calls", [])
                for tc in tool_calls:
                    args_preview = str(tc.get("args", {}))[:120]
                    print(f"\n🔧 {tc['name']}({args_preview})")
                if msg.content and not tool_calls:
                    print(f"\nAgent: {msg.content}")


def run_stream(supervisor, input_, config):
    """Run supervisor stream, print progress. Returns True if an interrupt is pending."""
    for chunk in supervisor.stream(input_, config=config, stream_mode="updates"):
        _print_stream_chunk(chunk)
    # After stream ends, check if graph is paused on an interrupt
    state = supervisor.get_state(config)
    return bool(state.interrupts)


def _extract_hitl_args(interrupts) -> tuple[str, str]:
    """Extract (filename, content) from HITLRequest stored in state.interrupts.

    Structure:
        state.interrupts[0].value       → HITLRequest TypedDict
        HITLRequest["action_requests"][0] → ActionRequest
        ActionRequest["args"]           → {"filename": ..., "content": ...}
    """
    try:
        hitl_request = interrupts[0].value          # HITLRequest TypedDict
        action_request = hitl_request["action_requests"][0]
        args = action_request["args"]
        return args.get("filename", "report.md"), args.get("content", "")
    except (IndexError, KeyError, TypeError, AttributeError):
        return "report.md", ""


def handle_hitl(supervisor, config: dict):
    """Show save_report preview and loop until approve/edit/reject."""
    state = supervisor.get_state(config)
    filename, content = _extract_hitl_args(state.interrupts)

    while True:
        print_separator()
        print("⏸️  ACTION REQUIRES APPROVAL")
        print_separator()
        print(f"  Tool:  save_report")
        print(f"  File:  {filename}")
        print(f"  Preview:")
        print()
        for line in content.splitlines()[:15]:
            print(f"    {line}")
        total_lines = len(content.splitlines())
        if total_lines > 15:
            print(f"    ... ({total_lines} lines total)")
        print()

        action = input("👉 approve / edit / reject: ").strip().lower()

        if action == "approve":
            result = supervisor.invoke(
                Command(resume={"decisions": [{"type": "approve"}]}),
                config=config,
            )
            print(f"\n✅ Approved! Report saved to output/{filename}")
            _print_final_message(result)
            return

        elif action == "edit":
            feedback = input("✏️  Your feedback: ").strip()
            # Reject with feedback: Supervisor sees the reason, reformulates the report,
            # and calls save_report again → triggers a new interrupt.
            interrupted = run_stream(
                supervisor,
                Command(resume={"decisions": [{"type": "reject", "message": feedback}]}),
                config,
            )
            if interrupted:
                # New save_report call pending → refresh args and loop
                state = supervisor.get_state(config)
                filename, content = _extract_hitl_args(state.interrupts)
                continue
            else:
                print("\n(Supervisor did not call save_report again after feedback.)")
                return

        elif action == "reject":
            result = supervisor.invoke(
                Command(resume={"decisions": [{"type": "reject", "message": "User rejected the report"}]}),
                config=config,
            )
            print("\n❌ Report saving cancelled.")
            _print_final_message(result)
            return

        else:
            print("Please enter 'approve', 'edit', or 'reject'.")


def _print_final_message(result):
    if not result or "messages" not in result:
        return
    last = result["messages"][-1]
    if hasattr(last, "content") and last.content and getattr(last, "type", None) == "ai":
        print(f"\nAgent: {last.content}")


def run_query(supervisor, user_input: str, config: dict):
    interrupted = run_stream(
        supervisor,
        {"messages": [{"role": "user", "content": user_input}]},
        config,
    )
    if interrupted:
        handle_hitl(supervisor, config)


def main():
    print("Research Multi-Agent System (type 'exit' to quit)")
    print("-" * 60)
    print("Supervisor → Planner → Researcher → Critic → HITL save")
    print("-" * 60)

    supervisor = create_supervisor()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        run_query(supervisor, user_input, config)


if __name__ == "__main__":
    main()
