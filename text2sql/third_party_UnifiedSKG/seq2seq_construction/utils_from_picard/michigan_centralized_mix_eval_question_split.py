import json
import numpy as np
from typing import Optional
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from seq2seq_construction.utils_from_picard.dataset import DataTrainingArguments, normalize, serialize_schema
from seq2seq_construction.utils_from_picard.trainer import Seq2SeqTrainer, EvalPrediction

from datasets.dataset_dict import DatasetDict
from torch.utils.data.dataset import T_co
import os
from tqdm import tqdm
from copy import deepcopy
import torch

def michigan_get_input(
    question: str,
    serialized_schema: str,
    prefix: str,
) -> str:
    return prefix + question.strip() + " " + serialized_schema.strip()


def michigan_get_target(
    query: str,
    db_id: str,
    normalize_query: bool,
    target_with_db_id: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)


def michigan_add_serialized_schema(ex: dict, data_training_args: DataTrainingArguments) -> dict:
    
    # import pdb
    # pdb.set_trace()
    
    serialized_schema = serialize_schema(
        question=ex["question"],
        db_path=ex["db_path"],
        db_id=ex["db_id"],
        db_column_names=ex["db_column_names"],
        db_table_names=ex["db_table_names"],
        schema_serialization_type=data_training_args.schema_serialization_type,
        schema_serialization_randomized=data_training_args.schema_serialization_randomized,
        schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
        schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
        normalize_query=data_training_args.normalize_query,
    )
    return {"serialized_schema": serialized_schema}


def michigan_pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    data_training_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else ""

    inputs = [
        michigan_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix)
        for question, serialized_schema in zip(batch["question"], batch["serialized_schema"])
    ]

    # model_inputs: dict = tokenizer(
    #     inputs,
    #     max_length=max_source_length,
    #     padding=False,
    #     truncation=True,
    #     return_overflowing_tokens=False,
    # )

    targets = [
        michigan_get_target(
            query=query,
            db_id=db_id,
            normalize_query=data_training_args.normalize_query,
            target_with_db_id=data_training_args.target_with_db_id,
        )
        for db_id, query in zip(batch["db_id"], batch["query"])
    ]

    # # Setup the tokenizer for targets
    # with tokenizer.as_target_tokenizer():
    #     labels = tokenizer(
    #         targets,
    #         max_length=max_target_length,
    #         padding=False,
    #         truncation=True,
    #         return_overflowing_tokens=False,
    #     )

    # model_inputs["labels"] = labels["input_ids"]
    # return model_inputs

    return zip(inputs, targets)

def michigan_pre_process_one_function(item: dict, args):
    prefix = ""

    seq_out = michigan_get_target(
        query=item["query"],
        db_id=item["db_id"],
        normalize_query=True,
        target_with_db_id=args.seq2seq.target_with_db_id,
    )

    return prefix + item["question"].strip(), seq_out


class MichiganTrainer(Seq2SeqTrainer):
    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
        label_ids = [f["labels"] for f in features]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
        metas = [
            {
                "query": x["query"],
                "question": x["question"],
                "context": context,            #####???
                "label": label,                #####???
                "db_id": x["db_id"],
                "db_path": x["db_path"],
                "db_table_names": x["db_table_names"],
                "db_column_names": x["db_column_names"],
                "db_foreign_keys": x["db_foreign_keys"],
            }
            for x, context, label in zip(examples, inputs, decoded_label_ids)
        ]
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        assert len(metas) == len(predictions)
        with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
            json.dump(
                [dict(**{"prediction": prediction}, **meta) for prediction, meta in zip(predictions, metas)],
                f,
                indent=4,
            )
        return EvalPrediction(predictions=predictions, label_ids=label_ids, metas=metas)

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids, metas = eval_prediction
        if self.target_with_db_id:
            # Remove database id from all predictions
            predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
        # TODO: using the decoded reference labels causes a crash in the spider evaluator
        # if self.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        # decoded_references = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # references = [{**{"query": r}, **m} for r, m in zip(decoded_references, metas)]
        references = metas
        self.metric.config_name = "test_suite"
        return self.metric.compute(predictions=predictions, references=references)

class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        # import pdb
        # pdb.set_trace()
        # if not len(raw_datasets) == 2:
        #     raise AssertionError("Train, Dev sections of dataset expected.")
        if getattr(self.args.seq2seq, "few_shot_rate"):
            raw_train = random.sample(list(raw_datasets["train"]), int(self.args.seq2seq.few_shot_rate * len(raw_datasets["train"])))
            train_dataset = TrainDataset(self.args, raw_train, cache_root)
        else:
            # import pdb
            # pdb.set_trace()
            train_dataset = TrainDataset(self.args, raw_datasets["train"], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets["validation"], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets["test"], cache_root)

        return train_dataset, dev_dataset, test_dataset


class TrainDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        question_split_data_types = ['question_split_geoquery_train.cache', 'question_split_scholar_train.cache', \
                                    'question_split_imdb_train.cache', 'question_split_yelp_train.cache', \
                                    'question_split_advising_train.cache', 'question_split_atis_train.cache', \
                                    'question_split_academic_train.cache', 'question_split_restaurants_train.cache']
        all_data = []
        self.seperate_train = {}
        for data_type in question_split_data_types:
            cache_path = os.path.join(cache_root, data_type)
            # import pdb
            # pdb.set_trace()
            if os.path.exists(cache_path) and args.dataset.use_cache:
                self.extended_data = torch.load(cache_path)
                all_data += self.extended_data
                self.seperate_train[data_type] = self.extended_data
            else:
                self.extended_data = []
                # for raw_data in tqdm(self.raw_datasets):
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = deepcopy(raw_data)
                    extend_data.update(michigan_add_serialized_schema(extend_data, args.seq2seq))

                    question, seq_out = michigan_pre_process_one_function(extend_data, args=self.args)
                    extend_data.update({"struct_in": extend_data["serialized_schema"].strip(),
                                        "text_in": question,
                                        "seq_out": seq_out})
                    self.extended_data.append(extend_data)
                all_data += self.extended_data
                self.seperate_train[data_type] = self.extended_data
                if args.dataset.use_cache:
                    torch.save(self.extended_data, cache_path)
        self.extended_data = all_data

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)

    def get_all_train_data(self):
        return self.seperate_train


class DevDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        question_split_data_types = ['question_split_geoquery_dev.cache', 'question_split_scholar_dev.cache', \
                                    'question_split_imdb_dev.cache', 'question_split_yelp_dev.cache', \
                                    'question_split_advising_dev.cache', 'question_split_atis_dev.cache', \
                                    'question_split_academic_dev.cache', 'question_split_restaurants_dev.cache']
        
        
        all_data = []
        self.seperate_dev = {}
        for data_type in question_split_data_types:
            cache_path = os.path.join(cache_root, data_type)
            if os.path.exists(cache_path) and args.dataset.use_cache:
                self.extended_data = torch.load(cache_path)
                all_data += self.extended_data
                self.seperate_dev[data_type] = self.extended_data
            else:
                self.extended_data = []
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = deepcopy(raw_data)
                    extend_data.update(michigan_add_serialized_schema(extend_data, args.seq2seq))

                    question, seq_out = michigan_pre_process_one_function(extend_data, args=self.args)
                    extend_data.update({"struct_in": extend_data["serialized_schema"].strip(),
                                        "text_in": question,
                                        "seq_out": seq_out})
                    self.extended_data.append(extend_data)
                all_data += self.extended_data
                self.seperate_dev[data_type] = self.extended_data
                if args.dataset.use_cache:
                    torch.save(self.extended_data, cache_path)
        self.extended_data = all_data

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
    
    def get_all_dev_data(self):
        return self.seperate_dev

class TestDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        question_split_data_types = ['question_split_geoquery_test.cache', 'question_split_scholar_test.cache', \
                                    'question_split_imdb_test.cache', 'question_split_yelp_test.cache', \
                                    'question_split_advising_test.cache', 'question_split_atis_test.cache', \
                                    'question_split_academic_test.cache', 'question_split_restaurants_test.cache']
        all_data = []
        self.seperate_test = {}
        for data_type in question_split_data_types:
            cache_path = os.path.join(cache_root, data_type)
            if os.path.exists(cache_path) and args.dataset.use_cache:
                self.extended_data = torch.load(cache_path)
                all_data += self.extended_data
                self.seperate_test[data_type] = self.extended_data
            else:
                self.extended_data = []
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = deepcopy(raw_data)
                    extend_data.update(michigan_add_serialized_schema(extend_data, args.seq2seq))

                    question, seq_out = michigan_pre_process_one_function(extend_data, args=self.args)
                    extend_data.update({"struct_in": extend_data["serialized_schema"].strip(),
                                        "text_in": question,
                                        "seq_out": seq_out})
                    self.extended_data.append(extend_data)
                
                all_data += self.extended_data
                self.seperate_test[data_type] = self.extended_data
                
                if args.dataset.use_cache:
                    torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
    
    def get_all_test_data(self):
        return self.seperate_test