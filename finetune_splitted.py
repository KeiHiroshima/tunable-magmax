import warnings

import wandb
from src.args import parse_arguments
from src.backends.registry import resolve_backend
from src.utils import setup_logging

warnings.simplefilter("ignore")

if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logging(level=args.logger_mode)

    wandb.init(
        project="tunable_magmax",
        entity=args.wandb_entity_name,
        config=vars(args),
        name=f"{args.model}_{args.dataset}_finetune",
    )

    resolve_backend(args.dataset).finetune(args)

    wandb.finish()
