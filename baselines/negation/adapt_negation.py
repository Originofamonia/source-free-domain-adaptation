# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import dataclasses
import logging
import os
from os.path import basename, dirname
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Union
from filelock import FileLock
import time

from enum import Enum

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
# from torch.utils.data.dataset import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers.data.processors.utils import InputFeatures
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.data.metrics import acc_and_f1
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers.data.processors.glue import glue_convert_examples_to_features

# from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from train import (AdversarialNetworkDann, dann_adapt, cdan_adapt, 
    AdversarialNetworkCdan, tent_adapt, BatchNorm)
from run_negation import NegationDataset
import tent


logger = logging.getLogger(__name__)

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    tgt = "tgt"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # Only allowed task is Negation, don't need this field from Glue
    #task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        default=f'practice_text/negation',
        # dev is for test, train is for adaptation
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    data_file: str = field(
        default=f'practice_text/negation/dev.tsv',
        # dev is for test, train is for adaptation
        metadata={"help": "The input data file. Should be a .tsv with one instance per line."}
    )


class NegationProcessor(DataProcessor):
    """ Processor for the sdfa shared task negation datasets """
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        right_list = self._read_tsv(os.path.join(data_dir, "dev.tsv"))
        left_list = self._read_tsv(os.path.join(data_dir, "dev_labels.txt"))
        merged_list = [(l[0], r[0]) for l, r in zip(left_list, right_list)]
        # for left, right in zip(left_list, right_list):
        #     print(left, right)
        return self._create_examples(merged_list, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "test")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train2.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["-1", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if test_mode:
                text_a = line[0]
                label = None
            else:
                # flip the signs so that 1 is negated, that way the f1 calculation is automatically
                # the f1 score for the negated label.
                label = str( -1 * int(line[0]) )
                text_a = '\t'.join(line[1:])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class NegationDatasetAdapt(Dataset):
    """ Copy-pasted from GlueDataset with glue task-specific code changed
        moved into here to be self-contained
    """
    args: DataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = NegationProcessor()
        self.output_mode = 'classification'
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        dataset = basename(dirname(args.data_dir)) if args.data_dir[-1] == '/' else basename(args.data_dir)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_polarity_{}_{}_{}_{}".format(
                dataset, mode.value, tokenizer.__class__.__name__, str(args.max_seq_length),
            ),
        )
        label_list = self.processor.get_labels()
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.tgt:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return np.asarray(self.features[i].input_ids), np.asarray(self.features[i].attention_mask), # self.features[i].token_type_ids

    def get_labels(self):
        return self.label_list


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="tmills/roberta_sfda_sharpseed",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    input_dim: int = field(
        default=768,
        metadata={"help": "input_dim"}
    )
    hidden_dim: int = field(
        default=3072,
        metadata={"help": "hidden_dim"}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default=f'practice_data/res/negation',
        metadata={"help": "The output directory where the model predictions and will be written."}
    )
    logging_dir: str = field(
        default=f'outputs',
        metadata={"help": "logging_dir"}
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "train.tsv: target domain for unlabeled adaptation"}
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "dev.tsv labeled"}
    )
    do_predict: bool = field(
        default=True,
        metadata={"help": "do test"}
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "overwrite_output_dir"}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "DDP local rank"}
    )
    device: int = field(
        default=1,
        metadata={"help": "device id"}
    )
    n_gpu: int = field(
        default=1,
        metadata={"help": "n_gpu"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "fp16"}
    )
    seed: int = field(
        default=444,
        metadata={"help": "seed"}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "train_batch_size"}
    )
    dataloader_drop_last: bool = field(
        default=True,
        metadata={"help": "dataloader_drop_last"}
    )
    max_steps: int = field(
        default=1e2,
        metadata={"help": "max_steps"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "gradient_accumulation_steps"}
    )
    weight_decay: float = field(
        default=1e-4,
        metadata={"help": "weight_decay"}
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "learning_rate"}
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "adam_epsilon"}
    )
    warmup_steps: int = field(
        default=1e1,
        metadata={"help": "warmup_steps"}
    )
    num_epochs: int = field(
        default=3,
        metadata={"help": "num_epochs"}
    )
    kd: bool = field(
        default=True,
        metadata={"help": "KD loss"}
    )
    ent: bool = field(
        default=True,
        metadata={"help": "IM loss"}
    )
    gent: bool = field(
        default=True,
        metadata={"help": "gent IM loss"}
    )
    temperature: int = field(
        default=20,
        metadata={"help": "KD temperature"}
    )
    ent_par: int = field(
        default=1,
        metadata={"help": "ent_par IM loss"}
    )
    log_step: int = field(
        default=9,
        metadata={"help": "log_step"}
    )
    optim_method: str = field(
        default="Adam",
        metadata={"help": "optim_method"}
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

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = 2
        model_args.num_labels = num_labels
        output_mode = 'classification'
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task='negation',
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        additional_special_tokens=['<e>', '</e>']
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))  # src model
    
    # Get datasets
    # train_dataset = (  # this is labeled test dataset, for inference only
    #     NegationDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    # )
    test_dataset = NegationDataset.from_tsv(data_args.data_file, tokenizer)
    tgt_dataset = (  # dev is unlabeled target dataset
        NegationDatasetAdapt(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    src_dataset = (  # use this as src dataset
        NegationDatasetAdapt(data_args, tokenizer=tokenizer, mode="tgt", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )
    src_loader = DataLoader(src_dataset, shuffle=True, batch_size=training_args.batch_size)
    tgt_loader = DataLoader(tgt_dataset, shuffle=True, batch_size=training_args.batch_size)
    # eval_loader = DataLoader()

    def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            return acc_and_f1(preds, p.label_ids)

        return compute_metrics_fn
    
    model = model.to(training_args.device)
    # discriminator = AdversarialNetworkDann(model_args).to(training_args.device)
    discriminator = AdversarialNetworkCdan(model_args).to(training_args.device)
    tgt_encoder = deepcopy(model.roberta).to(training_args.device)
    classifier = model.classifier.to(training_args.device)

    for params in model.roberta.parameters():
        params.requires_grad = False

    # tgt_encoder_adapted = dann_adapt(training_args, model.roberta, tgt_encoder, 
    #     discriminator, classifier, src_loader, tgt_loader)
    # tgt_encoder_adapted = cdan_adapt(training_args, model.roberta, tgt_encoder, 
    #     discriminator, classifier, src_loader, tgt_loader)

    # adapted_model = deepcopy(model)
    # adapted_model.roberta = deepcopy(tgt_encoder_adapted)

    batchnorm = BatchNorm(128)
    combined_model = tent.CombinedModel(model.roberta, batchnorm, model.classifier)
    adapted_model = tent_adapt(training_args, combined_model)

    trainer = Trainer(
        model=adapted_model,
        args=TrainingArguments('save_run/'),
        compute_metrics=None,
    )

    predictions = trainer.predict(test_dataset=test_dataset).predictions
    predictions = np.argmax(predictions, axis=1)
    os.makedirs(training_args.output_dir, exist_ok=True)
    output_test_file = os.path.join(training_args.output_dir, 'system.tsv')

    with open(output_test_file, "w") as writer:
        logger.info("***** Test results *****")
        for index, item in enumerate(predictions):
            item = src_dataset.get_labels()[item]
            writer.write("%s\n" % item)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
