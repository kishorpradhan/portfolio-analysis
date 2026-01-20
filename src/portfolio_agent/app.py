import argparse

from portfolio_agent.config.settings import get_settings
from portfolio_agent.utils.logger import get_logger
from portfolio_agent.workflows.portfolio_analysis import (
    answer_question,
    run_codegen_only,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run portfolio analysis agent")
    parser.add_argument(
        "--question",
        default="How have i invested and sold every month and year",
        help="Question to analyze.",
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Only generate+execute code step; skip search/synthesis.",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip vector cache lookup and write.",
    )
    args = parser.parse_args()

    settings = get_settings()
    logger = get_logger(__name__)
    logger.info("Starting portfolio agent. Data dir: %s", settings.data_dir)

    if args.code_only:
        print("=== Generate & Execute Python Code ===")
        result = run_codegen_only(args.question, skip_cache=args.skip_cache)
        print(result["code"])
        print("\n=== Execution ===")
        print(result["execution"])
        return

    print("\n=== Full Analysis (planner + prep + codegen + sandbox + summary) ===")
    try:
        answer = answer_question(args.question, skip_cache=args.skip_cache)
    except Exception as exc:
        answer = f"Failed to run full flow: {exc}"
    print(answer)


if __name__ == "__main__":
    main()
