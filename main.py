from __future__ import annotations

from agent.agent import run_agent


def main() -> None:
    print("BrowserBot CLI. Type /bye to exit.")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"/bye", "/exit", "/quit"}:
            print("Bye.")
            break

        try:
            response = run_agent(query)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print("\n" + response)


if __name__ == "__main__":
    main()
