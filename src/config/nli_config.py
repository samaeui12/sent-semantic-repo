import argparse
from utils import get_model_argparse
from model import MODEL_MAPPING_DICT

def nli_parser_model_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        default='/app/data/open_data/KorNLI',
        #required=True,
        type=str,
        help="The input training data file."
    )
    parser.add_argument(
        "--valid_file",
        default='/app/data/open_data/KorSTS',
        #required=True,
        type=str,
        help="The input training data file.",
    )

    parser.add_argument(
        "--experiments_path",
        type=str,
        default='experiments/experiment.csv',
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--metric",
        default='spearman',
        type=str,
        help="The input training data file."
    )

    parser.add_argument(
        "--train_batch_size",
        default=256,
        type=int,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=64,
        type=int,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--model_max_len", default=512, type=int, help="Maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_percent",
        default=0.01,
        type=float,
        help="Percentage of linear warmup over warmup_steps.",
    )
    parser.add_argument(
        "--logging_steps",
        default=1000,
        type=int,
        help="Generate log during training at each logging step.",
    )
    parser.add_argument(
        "--steps_per_evaluate",
        default=100,
        type=int,
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--model_type",
        default='sent_roberta',
        type=str,
        help="must select model type in [{}]".format(", ".join(MODEL_MAPPING_DICT)),
    )
    parser.add_argument(
        "--data_type",
        default='nli',
        type=str,
    )
    parser.add_argument(
        "--margin",
        default='0.5',
        type=float,
    )
    parser.add_argument(
        "--loss",
        default='MultipleNegativesRankingLoss',
        type=str,
    )
    parser.add_argument(
        "--pretrained_model",
        default='klue/roberta-large',
        type=str,
        help="If there is pretrained Model",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--amp_use",
        default=False,
        type=bool,
        help="use amp or not "
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.05, 
        help="For SimCSE temperature"
    )
    parser.add_argument(
        "--patience_limit", 
        type=int, 
        default=3, 
        help="patience limit for early stopping"
    )
    parser = get_model_argparse(parser)
    args = parser.parse_args(args=[])
    #args = parser.parse_args()
    return args