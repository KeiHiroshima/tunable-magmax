import warnings

from src.args import parse_arguments
from src.backends.registry import resolve_backend
from src.utils import setup_logging

warnings.simplefilter("ignore")

if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logging(level=args.logger_mode)
    resolve_backend(args.dataset).merge_and_evaluate(args)
