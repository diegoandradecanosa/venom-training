# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union


@dataclass
class DVCDatasetTrainingArguments:
    """
    Arguments for training using DVC
    """

    dvc_data_repository: Optional[str] = field(
        default=None,
        metadata={"help": "Path to repository used for dvc_dataset_path"},
    )


@dataclass
class CustomDataTrainingArguments(DVCDatasetTrainingArguments):
    """
    Arguments for training using custom datasets
    """

    dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the custom dataset. Supports json, csv, dvc. "
                "For DVC, the to dvc dataset to load, of format dvc://path. "
                "For csv or json, the path containing the dataset. "
            ),
        },
    )

    text_column: Optional[str] = field(
        default="text",
        metadata={"help": "For custom datasets only. The text field key"},
    )

    remove_columns: Union[None, str, List] = field(
        default=None,
        metadata={"help": "Column names to remove after preprocessing custom datasets"},
    )

    preprocessing_func: Optional[Callable] = field(
        default=None, metadata={"help": "The preprcessing function to apply"}
    )


@dataclass
class DataTrainingArguments(CustomDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval

    Using `HfArgumentParser` we can turn this class into argparse
    arguments to be able to specify them on the command line
    """

    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the dataset to use (via the datasets library). "
                "Supports input as a string or DatasetDict from HF"
            )
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": ("The configuration name of the dataset to use"),
        },
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. "
            "Sequences longer  than this will be truncated, sequences shorter will "
            "be padded."
        },
    )
    concatenate_data: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to concatenate datapoints to fill max_seq_length"
        },
    )
    raw_kwargs: Optional[Dict] = field(
        default=None,
        metadata={"help": "Additional keyboard args to pass to datasets load_data"},
    )
    splits: Union[None, str, List, Dict] = field(
        default=None,
        metadata={"help": "Optional percentages of each split to download"},
    )
    num_calibration_samples: Optional[int] = field(
        default=512,
        metadata={"help": "Number of samples to use for one-shot calibration"},
    )
    streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "True to stream data from a cloud dataset"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. If False, "
            "will pad the samples dynamically when batching to the maximum length "
            "in the batch (which can be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number "
            "of training examples to this value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number "
            "of evaluation examples to this value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of "
                "prediction examples to this value if set."
            ),
        },
    )
