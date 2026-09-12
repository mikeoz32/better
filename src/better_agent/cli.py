"""Command-line entry point for the Better Agent harness."""

import argparse


def main() -> int:
    """Run the minimal Better Agent command-line entry point."""
    parser = argparse.ArgumentParser(
        prog="ba",
        description="Better Agent event-command coding-agent harness.",
    )
    parser.parse_args()
    return 0
