import logging
import os
import pdb
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, DownloadConfig
from torch.utils.data import DataLoader
import torch
from typing import Optional, Union

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    DataCollatorWithPadding,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

glue_tasks = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    'imdb': ("text", None),
    'boolq': ("passage", "question"),
    "sst5" : ("sentence", None),
    'yelp_polarity': ("text", None),
    'yelp_review_full': ("text", None),
    'ag_news': ("text", None),
    'race-high': ("article", "question"),
    'race-middle': ("article", "question"),
    'race-all': ("article", "question"),
    'dream': ("dialogue", "question"),
}

logger = logging.getLogger(__name__)

class DataCollatorForGPT:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    def __init__(self, tokenizer, dataset="race"):
        self.tokenizer = tokenizer
        self.dataset = dataset
    def __call__(self, features):
        input_ids = []
        attention_mask = []
        labels = []
        max_length = max([len(feat['input_ids']) for feat in features])
        for feat in features:
            input_ids.append(feat['input_ids'] + [feat['input_ids'][0]] *
                             (max_length - len(feat['input_ids'])))
            attention_mask.append(feat['attention_mask'] + [0] *
                             (max_length - len(feat['attention_mask'])))
            labels.append(feat['labels'] + [-100] *
                             (max_length - len(feat['labels'])))
        batch = {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'labels': torch.LongTensor(labels)
        }
        return batch

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    just_inference: bool = field(default=False, metadata={"help": "Just do inference to get outputs."})
    inf_strategy: Optional[str] = field(
        default="standard",
    )
    noise_strategy: Optional[str] = field(
        default="eda",
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    ablation_sr: bool = field(default=False, metadata={"help": "Ablation sr."})
    ablation_ri: bool = field(default=False, metadata={"help": "Ablation ri."})
    ablation_rs: bool = field(default=False, metadata={"help": "Ablation rs."})
    ablation_rd: bool = field(default=False, metadata={"help": "Ablation rd."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json", "tsv"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    teacher_num_layers: int = field(
        default=0,
        metadata={"help": "Number of teacher layers"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    download_config = DownloadConfig()
    download_config.use_etag = False
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)


    if data_args.task_name is not None and data_args.task_name in glue_tasks:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name, download_config=download_config)
    elif data_args.task_name is not None and data_args.task_name in task_to_keys and data_args.task_name != 'sst5':  # other supported tasks
        if "race" in data_args.task_name:
            datasets = load_dataset("race", data_args.task_name.split("-")[-1], download_config=download_config)
        elif data_args.task_name == "dream":
            datasets = load_dataset("dream", download_config=download_config)
        else:
            datasets = load_dataset(data_args.task_name, download_config=download_config)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                        test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files, download_config=download_config)
        elif data_args.train_file.endswith(".tsv"):
            datasets = load_dataset("csv", data_files=data_files, delimiter="\t", download_config=download_config)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files, download_config=download_config)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None and data_args.task_name in glue_tasks:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    elif data_args.task_name is not None and "race" in data_args.task_name:
        is_regression = False
        label_list = ["A", "B", "C", "D"]
        num_labels = 4
    elif data_args.task_name is not None and data_args.task_name == "dream":
        is_regression = False
        label_list = ["A", "B", "C"]
        num_labels = 3
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        elif data_args.task_name == 'boolq':
            label_list = ["False", "True"]
            num_labels = 2
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    logger.info("Number of Label :%d" % num_labels)
    # print(label_list)
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.teacher_num_layers > 0:
        config.num_hidden_layers = model_args.teacher_num_layers
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = GPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression or data_args.task_name == 'sst5':
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        NOISE_TIMES=10
        if data_args.inf_strategy == "standard" or data_args.inf_strategy == "surrogate":
            pass
        elif data_args.inf_strategy == "noise":
            if data_args.noise_strategy == "eda":
                from utils.add_noise_eda import add_noise_to_all
            else:
                assert False

            if data_args.noise_strategy == "eda":
                DEFAULT_RATIO = 0.1
                examples[sentence1_key] = add_noise_to_all(examples[sentence1_key], NOISE_TIMES,
                                                           alpha_sr=0 if data_args.ablation_sr else DEFAULT_RATIO ,
                                                           alpha_ri=0 if data_args.ablation_ri else DEFAULT_RATIO,
                                                           alpha_rs=0 if data_args.ablation_rs else DEFAULT_RATIO,
                                                           p_rd=0 if data_args.ablation_rd else DEFAULT_RATIO)
                if sentence2_key is not None:
                    examples[sentence2_key] = add_noise_to_all(examples[sentence2_key], NOISE_TIMES,
                                                           alpha_sr=0 if data_args.ablation_sr else DEFAULT_RATIO ,
                                                           alpha_ri=0 if data_args.ablation_ri else DEFAULT_RATIO,
                                                           alpha_rs=0 if data_args.ablation_rs else DEFAULT_RATIO,
                                                           p_rd=0 if data_args.ablation_rd else DEFAULT_RATIO)
            else:
                examples[sentence1_key] = add_noise_to_all(examples[sentence1_key], NOISE_TIMES)
                if sentence2_key is not None:
                    examples[sentence2_key] = add_noise_to_all(examples[sentence2_key], NOISE_TIMES)

            if "race" in data_args.task_name:
                # examples['options'] = [item for item in sum(examples["options"], []) for i in range(NOISE_TIMES)]
                examples["options"] = add_noise_to_all(sum(examples["options"], []), NOISE_TIMES)
                examples["options"] = [[examples["options"][i+j: i+j + 4*NOISE_TIMES: NOISE_TIMES] for j in range(NOISE_TIMES)] for i in range(0, len(examples["options"]), 4*NOISE_TIMES)]
                examples["options"] = sum(examples["options"], [])
                for key in ['idx', 'answer']:
                    if key in examples:
                        examples[key] = [item for item in examples[key] for i in range(NOISE_TIMES)]
            elif data_args.task_name == "dream":
                # examples["answer"] = [examples["choice"][i].index(ans) for i, ans in enumerate(examples["answer"])]
                examples["choice"] = add_noise_to_all(sum(examples["choice"], []), NOISE_TIMES)
                examples['choice'] = [item for item in sum(examples["choice"], []) for i in range(NOISE_TIMES)]
                examples["choice"] = [
                    [examples["choice"][i + j: i + j + 3 * NOISE_TIMES: NOISE_TIMES] for j in range(NOISE_TIMES)] for i
                    in range(0, len(examples["choice"]), 3 * NOISE_TIMES)]
                examples["choice"] = sum(examples["choice"], [])
                for key in ['idx', 'answer']:
                    if key in examples:
                        examples[key] = [item for item in examples[key] for i in range(NOISE_TIMES)]
            else:
                for key in ['idx', 'label']:
                    if key in examples:
                        examples[key] = [item for item in examples[key] for i in range(NOISE_TIMES)]
        else:
            assert False

        if "race" in data_args.task_name:
            context = examples[sentence1_key]
            question = examples[sentence2_key]
            options = examples["options"]
            prompt = [f"{tokenizer.bos_token}Context: {c} Question: {q} Options: A. {o[0]} B. {o[1]} " \
                      f"C. {o[2]} D. {o[3]} Answer:" for c, q, o in zip(context, question, options)]
            result = tokenizer(prompt, max_length=max_seq_length, truncation=True, add_special_tokens=True)
            answer = [f" {lb}" for lb in examples['answer']]
            answer = tokenizer(answer)
            result["labels"] = [[-100] * (len(input_ids) - 1) + ans for input_ids, ans in
                                zip(result['input_ids'], answer['input_ids'])]
        else:
            assert False
        return result

    if 'idx' not in datasets["train"].features:
        idx_column = [i for i in range(len(datasets["train"]))]
        datasets["train"] = datasets["train"].add_column("idx", idx_column)

    # pdb.set_trace()
    train_dataset = datasets["train"]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    train_dataset = train_dataset.map(preprocess_function, batched=True,
                            load_from_cache_file=not data_args.overwrite_cache, remove_columns=datasets["test"].column_names)

    logger.info("*** Inference ***")
    if "train" not in datasets:
        raise ValueError("--do_train requires a train dataset")
    model.cuda()
    model.eval()
    ans_list = []
    idx_list = []
    logits_list = []
    choice_idx = [tokenizer.encode(choice)[0] for choice in [' A', ' B', ' C', ' D']]
    if 'idx' not in train_dataset.features:
        idx_column = [i for i in range(len(train_dataset))]
        train_dataset = train_dataset.add_column("idx", idx_column)
    with torch.no_grad():
        from tqdm import tqdm
        for i in tqdm(range(len(train_dataset))):
            idx = train_dataset[i]['idx']
            inp = {k: torch.LongTensor(v).cuda() for k, v in train_dataset[i].items() if k != 'idx'}
            pred = model(**inp)
            logits_list.append(pred['logits'][-1][choice_idx].cpu())
            ans_list.append(pred['logits'][-1].argmax())
            idx_list.append(idx)
    logits_list = torch.stack(logits_list)
    idx_list = torch.LongTensor(idx_list)
    ans_list = torch.LongTensor(ans_list)
    torch.save(logits_list, os.path.join(training_args.output_dir, "logits_list.pt"))
    torch.save(ans_list, os.path.join(training_args.output_dir, "ans_list.pt"))
    torch.save(idx_list, os.path.join(training_args.output_dir, "idx_list.pt"))
    torch.save(train_dataset, os.path.join(training_args.output_dir, "train_dataset.pt"))

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

