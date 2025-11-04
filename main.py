"""Main CLI entry point for PDF RAG system."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config
from src.indexing_pipeline import IndexingPipeline
from src.query_pipeline import QueryPipeline

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level.
    
    Args:
        verbose: If True, enable full logging with timestamps. Default is quiet mode.
    """
    if verbose:
        # Verbose mode: full logging with timestamps
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        # Quiet mode (default): no timestamps, only WARNING and ERROR
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )


def index_command(args):
    """Index PDF files."""
    # Index always uses verbose logging (old way)
    setup_logging(verbose=True)
    config = Config(config_path=args.config, env_path=args.env)
    pipeline = IndexingPipeline(config)

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error(f"Path does not exist: {pdf_path}")
        return 1

    if pdf_path.is_file():
        if pdf_path.suffix.lower() != ".pdf":
            logger.error(f"File is not a PDF: {pdf_path}")
            return 1
        success = pipeline.index_pdf(pdf_path)
        return 0 if success else 1
    elif pdf_path.is_dir():
        count = pipeline.index_directory(pdf_path)
        return 0 if count > 0 else 1
    else:
        logger.error(f"Invalid path: {pdf_path}")
        return 1


def query_command(args):
    """Query the RAG system."""
    # Check verbose flag from either main parser or subparser
    verbose = getattr(args, 'verbose', False)
    setup_logging(verbose=verbose)
    config = Config(config_path=args.config, env_path=args.env)
    pipeline = QueryPipeline(config)

    response = pipeline.query(args.query)
    print("\n" + "=" * 80)
    print("RESPONSE:")
    print("=" * 80)
    print(response)
    print("=" * 80 + "\n")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PDF RAG System - Index PDFs and answer questions"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.yaml file",
    )
    parser.add_argument(
        "--env",
        type=Path,
        help="Path to .env file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging with timestamps for query command (default is quiet mode)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Index PDF file(s)"
    )
    index_parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to PDF file or directory containing PDFs",
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query", help="Query the RAG system"
    )
    query_parser.add_argument(
        "query",
        type=str,
        help="Query to ask",
    )
    query_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging with timestamps for query (default is quiet mode)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "index":
        return index_command(args)
    elif args.command == "query":
        return query_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
