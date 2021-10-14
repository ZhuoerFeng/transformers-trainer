import sys
import os
import torch


from transformers import (
    HfArgumentParser,
    Trainer,
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
)

from arguments import DataTrainingArguments, ModelArguments

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args, data_args, training_args)


if __name__ == '__main__':
    main()