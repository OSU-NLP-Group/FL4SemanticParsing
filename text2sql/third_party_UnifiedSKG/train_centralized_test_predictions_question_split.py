import logging
import os
import time
import json

import torch
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from collections import OrderedDict
import utils.tool
from utils.configue import Configure
from utils.dataset import TokenizedDataset
# from utils.trainer import EvaluateFriendlySeq2SeqTrainerSetGPU
from utils.trainer_for_finetune_centralized import EvaluateFriendlySeq2SeqTrainer
from utils.training_arguments import WrappedSeq2SeqTrainingArguments
import sys
sys.path.append("..") 

# import os 
# os.environ['WANDB_API_KEY'] = ' '
# os.environ['WANDB_PROJEC'] = 'lorar_release'
# os.environ['WANDB_ENTIT'] = 'zts'

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Huggingface realized the "Seq2seqTrainingArguments" which is the same with "WrappedSeq2SeqTrainingArguments"
# in transformers==4.10.1 during our work.
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ[
        'CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Deterministic behavior of torch.addmm. Please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    
    # comment this line since upgrade the torch version in strawberry1 server
    # torch.set_deterministic(True)
    
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)

    from filelock import FileLock
    import nltk
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)

    # Get args
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    args = Configure.Get(training_args.cfg)
    # import pdb
    # pdb.set_trace()

    # add for strawberry1 server
    torch.cuda.set_device(training_args.local_rank)

    if 'checkpoint-???' in args.bert.location:
        args.bert.location = get_last_checkpoint(
            os.path.dirname(args.bert.location.model_name_or_path))
        
        logger.info(f"Resolve model_name_or_path to {args.bert.location.model_name_or_path}")

    if "wandb" in training_args.report_to and training_args.local_rank <= 0:
        import wandb

        init_args = {}
        if "MLFLOW_EXPERIMENT_ID" in os.environ:
            init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
        # wandb.init(
        #     project=os.getenv("WANDB_PROJECT", "uni-frame-for-knowledge-tabular-tasks"),
        #     name=training_args.run_name,
        #     entity=os.getenv("WANDB_ENTITY", 'sgtnew'),
        #     **init_args,
        # )

        #comment for debugging
        wandb.init(project="lorar_release", name=training_args.run_name, entity="zts", **init_args,)
        wandb.config.update(training_args, allow_val_change=True)

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    os.makedirs(training_args.output_dir, exist_ok=True)

    # The inputs will be train, dev, test or train, dev now.
    # We deprecate the k-fold cross-valid function since it causes too many avoidable troubles.

    # import pdb
    # pdb.set_trace()

    if not args.arg_paths:
        # cache_root = os.path.join('output', 'cache_backup')
        cache_root = '/home/zhang.11535/UnifiedSKG/output/cache_backup'
        os.makedirs(cache_root, exist_ok=True)
        raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(path=args.dataset.loader_path,
                                                                         cache_dir=args.dataset.data_store_path)
        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).to_seq2seq(
            raw_datasets_split, cache_root)

    else:
        # cache_root = os.path.join('output', 'cache_backup')
        cache_root = '/home/zhang.11535/UnifiedSKG/output/cache_backup'
        os.makedirs(cache_root, exist_ok=True)
        meta_tuning_data = {}
        for task, arg_path in args.arg_paths:
            # print(args.arg_paths)
            # print(arg_path)
            # import pdb
            # pdb.set_trace()

            task_args = Configure.Get(arg_path)
            task_args.bert = args.bert
            print('task_args.bert.location:', task_args.bert.location)
            task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(
                path=task_args.dataset.loader_path,
                cache_dir=task_args.dataset.data_store_path)
            task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args).\
                to_seq2seq(task_raw_datasets_split, cache_root)

            meta_tuning_data[arg_path] = task_seq2seq_dataset_split

        
        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(meta_tuning_data)


    evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
    model = utils.tool.get_model(args.model.name)(args)
    model_tokenizer = model.tokenizer


    seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = None, None, None
    if len(seq2seq_dataset_split) == 2:
        seq2seq_train_dataset, seq2seq_eval_dataset = seq2seq_dataset_split
    elif len(seq2seq_dataset_split) == 3:
        seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = seq2seq_dataset_split
    else:
        raise ValueError("Other split not support yet.")

    # import pdb
    # pdb.set_trace()

    # We wrap the "string" seq2seq data into "tokenized tensor".
    train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                     seq2seq_train_dataset) if seq2seq_train_dataset else None
    eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_eval_dataset) if seq2seq_eval_dataset else None
    test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_test_dataset) if seq2seq_test_dataset else None
    
    # print(task_seq2seq_dataset_split[0].__dict__.keys())
    # import pdb
    # pdb.set_trace()
    # train data

    question_split_geoquery_train = task_seq2seq_dataset_split[0].seperate_train["question_split_geoquery_train.cache"]
    question_split_scholar_train = task_seq2seq_dataset_split[0].seperate_train["question_split_scholar_train.cache"]
    question_split_imdb_train = task_seq2seq_dataset_split[0].seperate_train["question_split_imdb_train.cache"]
    question_split_yelp_train = task_seq2seq_dataset_split[0].seperate_train["question_split_yelp_train.cache"]
    question_split_advising_train = task_seq2seq_dataset_split[0].seperate_train["question_split_advising_train.cache"]
    question_split_atis_train = task_seq2seq_dataset_split[0].seperate_train["question_split_atis_train.cache"]
    question_split_academic_train = task_seq2seq_dataset_split[0].seperate_train["question_split_academic_train.cache"]
    question_split_restaurants_train = task_seq2seq_dataset_split[0].seperate_train["question_split_restaurants_train.cache"]

    question_split_geoquery_dev = task_seq2seq_dataset_split[1].seperate_dev["question_split_geoquery_dev.cache"]
    question_split_scholar_dev = task_seq2seq_dataset_split[1].seperate_dev["question_split_scholar_dev.cache"]
    question_split_imdb_dev = task_seq2seq_dataset_split[1].seperate_dev["question_split_imdb_dev.cache"]
    question_split_yelp_dev = task_seq2seq_dataset_split[1].seperate_dev["question_split_yelp_dev.cache"]
    question_split_advising_dev = task_seq2seq_dataset_split[1].seperate_dev["question_split_advising_dev.cache"]
    question_split_atis_dev = task_seq2seq_dataset_split[1].seperate_dev["question_split_atis_dev.cache"]
    question_split_academic_dev = task_seq2seq_dataset_split[1].seperate_dev["question_split_academic_dev.cache"]
    question_split_restaurants_dev = task_seq2seq_dataset_split[1].seperate_dev["question_split_restaurants_dev.cache"]

    question_split_geoquery_test = task_seq2seq_dataset_split[2].seperate_test["question_split_geoquery_test.cache"]
    question_split_scholar_test = task_seq2seq_dataset_split[2].seperate_test["question_split_scholar_test.cache"]
    question_split_imdb_test = task_seq2seq_dataset_split[2].seperate_test["question_split_imdb_test.cache"]
    question_split_yelp_test = task_seq2seq_dataset_split[2].seperate_test["question_split_yelp_test.cache"]
    question_split_advising_test = task_seq2seq_dataset_split[2].seperate_test["question_split_advising_test.cache"]
    question_split_atis_test = task_seq2seq_dataset_split[2].seperate_test["question_split_atis_test.cache"]
    question_split_academic_test = task_seq2seq_dataset_split[2].seperate_test["question_split_academic_test.cache"]
    question_split_restaurants_test = task_seq2seq_dataset_split[2].seperate_test["question_split_restaurants_test.cache"]

    
    question_split_geoquery_meta_tuning_data = {}
    question_split_geoquery_meta_tuning_data["META_TUNING/centralized_with_cell.cfg"] = (question_split_geoquery_train, question_split_geoquery_dev, question_split_geoquery_test)
    seq2seq_geoquery_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(question_split_geoquery_meta_tuning_data)

    question_split_scholar_meta_tuning_data = {}
    question_split_scholar_meta_tuning_data["META_TUNING/centralized_with_cell.cfg"] = (question_split_scholar_train, question_split_scholar_dev, question_split_scholar_test)
    seq2seq_scholar_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(question_split_scholar_meta_tuning_data)

    question_split_imdb_meta_tuning_data = {}
    question_split_imdb_meta_tuning_data["META_TUNING/centralized_with_cell.cfg"] = (question_split_imdb_train, question_split_imdb_dev, question_split_imdb_test)
    seq2seq_imdb_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(question_split_imdb_meta_tuning_data)

    question_split_yelp_meta_tuning_data = {}
    question_split_yelp_meta_tuning_data["META_TUNING/centralized_with_cell.cfg"] = (question_split_yelp_train, question_split_yelp_dev, question_split_yelp_test)
    seq2seq_yelp_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(question_split_yelp_meta_tuning_data)

    question_split_advising_meta_tuning_data = {}
    question_split_advising_meta_tuning_data["META_TUNING/centralized_with_cell.cfg"] = (question_split_advising_train, question_split_advising_dev, question_split_advising_test)
    seq2seq_advising_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(question_split_advising_meta_tuning_data)

    question_split_atis_meta_tuning_data = {}
    question_split_atis_meta_tuning_data["META_TUNING/centralized_with_cell.cfg"] = (question_split_atis_train, question_split_atis_dev, question_split_atis_test)
    seq2seq_atis_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(question_split_atis_meta_tuning_data)

    question_split_academic_meta_tuning_data = {}
    question_split_academic_meta_tuning_data["META_TUNING/centralized_with_cell.cfg"] = (question_split_academic_train, question_split_academic_dev, question_split_academic_test)
    seq2seq_academic_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(question_split_academic_meta_tuning_data)

    question_split_restaurants_meta_tuning_data = {}
    question_split_restaurants_meta_tuning_data["META_TUNING/centralized_with_cell.cfg"] = (question_split_restaurants_train, question_split_restaurants_dev, question_split_restaurants_test)
    seq2seq_restaurants_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(question_split_restaurants_meta_tuning_data)

    table_dict_path = "/home/zhang.11535/UnifiedSKG/eight_data/table_dict.json" 
    with open(table_dict_path , "r") as f:
        table_dict = json.load(f)
    for i in [0,1,2]:
        for j in range(len(seq2seq_geoquery_dataset_split[i])):
            seq2seq_geoquery_dataset_split[i][j]["serialized_schema"] = table_dict["serialized_schema"]["geoquery"]
            seq2seq_geoquery_dataset_split[i][j]["struct_in"] = table_dict["struct_in"]["geoquery"]

        for j in range(len(seq2seq_advising_dataset_split[i])):   
            seq2seq_advising_dataset_split[i][j]["serialized_schema"] = table_dict["serialized_schema"]["advising"]
            seq2seq_advising_dataset_split[i][j]["struct_in"] = table_dict["struct_in"]["advising"]
        
        for j in range(len(seq2seq_atis_dataset_split[i])):
            seq2seq_atis_dataset_split[i][j]["serialized_schema"] = table_dict["serialized_schema"]["atis"]
            seq2seq_atis_dataset_split[i][j]["struct_in"] = table_dict["struct_in"]["atis"]

        for j in range(len(seq2seq_restaurants_dataset_split[i])):    
            seq2seq_restaurants_dataset_split[i][j]["serialized_schema"] = table_dict["serialized_schema"]["restaurants"]
            seq2seq_restaurants_dataset_split[i][j]["struct_in"] = table_dict["struct_in"]["restaurants"]

        for j in range(len(seq2seq_scholar_dataset_split[i])):    
            seq2seq_scholar_dataset_split[i][j]["serialized_schema"] = table_dict["serialized_schema"]["scholar"]
            seq2seq_scholar_dataset_split[i][j]["struct_in"] = table_dict["struct_in"]["scholar"]

        for j in range(len(seq2seq_academic_dataset_split[i])):    
            seq2seq_academic_dataset_split[i][j]["serialized_schema"] = table_dict["serialized_schema"]["academic"]
            seq2seq_academic_dataset_split[i][j]["struct_in"] = table_dict["struct_in"]["academic"]

        for j in range(len(seq2seq_imdb_dataset_split[i])):    
            seq2seq_imdb_dataset_split[i][j]["serialized_schema"] = table_dict["serialized_schema"]["imdb"]
            seq2seq_imdb_dataset_split[i][j]["struct_in"] = table_dict["struct_in"]["imdb"]

        for j in range(len(seq2seq_yelp_dataset_split[i])):    
            seq2seq_yelp_dataset_split[i][j]["serialized_schema"] = table_dict["serialized_schema"]["yelp"]
            seq2seq_yelp_dataset_split[i][j]["struct_in"] = table_dict["struct_in"]["yelp"]
    
    
    seq2seq_question_split_eight_train_dataset = []  
    seq2seq_question_split_eight_train_dataset.append([seq2seq_geoquery_dataset_split[0]])
    seq2seq_question_split_eight_train_dataset.append([seq2seq_scholar_dataset_split[0]])
    seq2seq_question_split_eight_train_dataset.append([seq2seq_imdb_dataset_split[0]]) 
    seq2seq_question_split_eight_train_dataset.append([seq2seq_yelp_dataset_split[0]])
    seq2seq_question_split_eight_train_dataset.append([seq2seq_advising_dataset_split[0]])
    seq2seq_question_split_eight_train_dataset.append([seq2seq_atis_dataset_split[0]])
    seq2seq_question_split_eight_train_dataset.append([seq2seq_academic_dataset_split[0]])
    seq2seq_question_split_eight_train_dataset.append([seq2seq_restaurants_dataset_split[0]])

    question_split_geoquery_train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_geoquery_dataset_split[0]) if seq2seq_geoquery_dataset_split[0] else None
    question_split_scholar_train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_scholar_dataset_split[0]) if seq2seq_scholar_dataset_split[0] else None
    question_split_imdb_train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_imdb_dataset_split[0]) if seq2seq_imdb_dataset_split[0] else None
    question_split_yelp_train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_yelp_dataset_split[0]) if seq2seq_yelp_dataset_split[0] else None
    question_split_advising_train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_advising_dataset_split[0]) if seq2seq_advising_dataset_split[0] else None
    question_split_atis_train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_atis_dataset_split[0]) if seq2seq_atis_dataset_split[0] else None
    question_split_academic_train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_academic_dataset_split[0]) if seq2seq_academic_dataset_split[0] else None
    question_split_restaurants_train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_restaurants_dataset_split[0]) if seq2seq_restaurants_dataset_split[0] else None
    
    question_split_eight_train_dataset = []
    question_split_eight_train_dataset.append([question_split_geoquery_train_dataset])
    question_split_eight_train_dataset.append([question_split_scholar_train_dataset])
    question_split_eight_train_dataset.append([question_split_imdb_train_dataset])
    question_split_eight_train_dataset.append([question_split_yelp_train_dataset])
    question_split_eight_train_dataset.append([question_split_advising_train_dataset])
    question_split_eight_train_dataset.append([question_split_atis_train_dataset])
    question_split_eight_train_dataset.append([question_split_academic_train_dataset])
    question_split_eight_train_dataset.append([question_split_restaurants_train_dataset])

    # dev data
    seq2seq_question_split_eight_eval_dataset = []  
    seq2seq_question_split_eight_eval_dataset.append([seq2seq_geoquery_dataset_split[1]])
    seq2seq_question_split_eight_eval_dataset.append([seq2seq_scholar_dataset_split[1]])
    seq2seq_question_split_eight_eval_dataset.append([seq2seq_imdb_dataset_split[1]]) 
    seq2seq_question_split_eight_eval_dataset.append([seq2seq_yelp_dataset_split[1]])
    seq2seq_question_split_eight_eval_dataset.append([seq2seq_advising_dataset_split[1]])
    seq2seq_question_split_eight_eval_dataset.append([seq2seq_atis_dataset_split[1]])
    seq2seq_question_split_eight_eval_dataset.append([seq2seq_academic_dataset_split[1]])
    seq2seq_question_split_eight_eval_dataset.append([seq2seq_restaurants_dataset_split[1]])

    question_split_geoquery_eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_geoquery_dataset_split[1]) if seq2seq_geoquery_dataset_split[1] else None
    question_split_scholar_eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_scholar_dataset_split[1]) if seq2seq_scholar_dataset_split[1] else None
    question_split_imdb_eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_imdb_dataset_split[1]) if seq2seq_imdb_dataset_split[1] else None
    question_split_yelp_eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_yelp_dataset_split[1]) if seq2seq_yelp_dataset_split[1] else None
    question_split_advising_eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_advising_dataset_split[1]) if seq2seq_advising_dataset_split[1] else None
    question_split_atis_eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_atis_dataset_split[1]) if seq2seq_atis_dataset_split[1] else None
    question_split_academic_eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_academic_dataset_split[1]) if seq2seq_academic_dataset_split[1] else None
    question_split_restaurants_eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_restaurants_dataset_split[1]) if seq2seq_restaurants_dataset_split[1] else None
    

    question_split_eight_eval_dataset = []
    question_split_eight_eval_dataset.append([question_split_geoquery_eval_dataset])
    question_split_eight_eval_dataset.append([question_split_scholar_eval_dataset])
    question_split_eight_eval_dataset.append([question_split_imdb_eval_dataset])
    question_split_eight_eval_dataset.append([question_split_yelp_eval_dataset])
    question_split_eight_eval_dataset.append([question_split_advising_eval_dataset])
    question_split_eight_eval_dataset.append([question_split_atis_eval_dataset])
    question_split_eight_eval_dataset.append([question_split_academic_eval_dataset])
    question_split_eight_eval_dataset.append([question_split_restaurants_eval_dataset])


    # test data
    
    seq2seq_question_split_eight_test_dataset = []  
    seq2seq_question_split_eight_test_dataset.append([seq2seq_geoquery_dataset_split[2]])
    seq2seq_question_split_eight_test_dataset.append([seq2seq_scholar_dataset_split[2]])
    seq2seq_question_split_eight_test_dataset.append([seq2seq_imdb_dataset_split[2]]) 
    seq2seq_question_split_eight_test_dataset.append([seq2seq_yelp_dataset_split[2]])
    seq2seq_question_split_eight_test_dataset.append([seq2seq_advising_dataset_split[2]])
    seq2seq_question_split_eight_test_dataset.append([seq2seq_atis_dataset_split[2]])
    seq2seq_question_split_eight_test_dataset.append([seq2seq_academic_dataset_split[2]])
    seq2seq_question_split_eight_test_dataset.append([seq2seq_restaurants_dataset_split[2]])

    question_split_geoquery_test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_geoquery_dataset_split[2]) if seq2seq_geoquery_dataset_split[2] else None
    question_split_scholar_test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_scholar_dataset_split[2]) if seq2seq_scholar_dataset_split[2] else None
    question_split_imdb_test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_imdb_dataset_split[2]) if seq2seq_imdb_dataset_split[2] else None
    question_split_yelp_test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_yelp_dataset_split[2]) if seq2seq_yelp_dataset_split[2] else None
    question_split_advising_test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_advising_dataset_split[2]) if seq2seq_advising_dataset_split[2] else None
    question_split_atis_test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_atis_dataset_split[2]) if seq2seq_atis_dataset_split[2] else None
    question_split_academic_test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_academic_dataset_split[2]) if seq2seq_academic_dataset_split[2] else None
    question_split_restaurants_test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_restaurants_dataset_split[2]) if seq2seq_restaurants_dataset_split[2] else None
    
    question_split_eight_test_dataset = []
    question_split_eight_test_dataset.append([question_split_geoquery_test_dataset])
    question_split_eight_test_dataset.append([question_split_scholar_test_dataset])
    question_split_eight_test_dataset.append([question_split_imdb_test_dataset])
    question_split_eight_test_dataset.append([question_split_yelp_test_dataset])
    question_split_eight_test_dataset.append([question_split_advising_test_dataset])
    question_split_eight_test_dataset.append([question_split_atis_test_dataset])
    question_split_eight_test_dataset.append([question_split_academic_test_dataset])
    question_split_eight_test_dataset.append([question_split_restaurants_test_dataset])

    # Initialize our Trainer
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.seq2seq.patience if args.seq2seq.patience else 12) ##original paper using 5, we change it to a larger one
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')
    # trainer = EvaluateFriendlySeq2SeqTrainerSetGPU(
    #     fl_algorithm = " ",
    #     fedprox_mu = 0,
    #     device = device,
    trainer = EvaluateFriendlySeq2SeqTrainer(
        args=training_args,
        model=model,
        evaluator=evaluator,
        # We name it "evaluator" while the hugging face call it "Metric",
        # they are all f(predictions: List, references: List of dict) = eval_result: dict
        tokenizer=model_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=seq2seq_eval_dataset,
        wandb_run_dir=wandb.run.dir if "wandb" in training_args.report_to and training_args.local_rank <= 0 else None,
        callbacks=[early_stopping_callback],
    )
    print('Trainer build successfully.')

    # Load model weights (for --do_train=False or post finetuning).
    if training_args.load_weights_from:
        print("loading weights from checkpoints...")
        state_dict = torch.load(os.path.join(training_args.load_weights_from, transformers.WEIGHTS_NAME), map_location="cpu")
        trainer.model.load_state_dict(state_dict, strict=True)
        # release memory
        del state_dict

    if args.load_multiple_prefix_module_weights_from:
        reconstruct_state_dict = OrderedDict()

        # load prefix modules
        for task_name, module_weight_location in args.load_multiple_prefix_module_weights_from:
            state_dict = torch.load(os.path.join(module_weight_location, transformers.WEIGHTS_NAME), map_location="cpu")
            MULTI_PREFIX_ATTR_NAME = "multi_prefix"
            for weight_name, stored_tensor in state_dict.items():
                if str(weight_name).startswith("pretrain_model"):
                    continue  # skip the pretrained model and we will load a new one from another place
                reconstruct_state_dict['{}.{}.{}'.format(MULTI_PREFIX_ATTR_NAME, "_".join(task_name.split("_")[:-1]), weight_name)] = stored_tensor
                # extract the prefix part and add them to dict

        # give it into the model
        trainer.model.load_state_dict(reconstruct_state_dict, strict=False)

        # release memory
        del reconstruct_state_dict

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # # Evaluation
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")

    #     metrics = trainer.evaluate(
    #         metric_key_prefix="eval"
    #     )
    #     max_eval_samples = len(eval_dataset)
    #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # # Predict
    # if training_args.do_predict:
    #     logger.info("*** Predict ***")

    #     predict_results = trainer.predict(
    #         test_dataset=test_dataset if test_dataset else eval_dataset,
    #         test_examples=seq2seq_test_dataset if seq2seq_test_dataset else seq2seq_eval_dataset,
    #         metric_key_prefix="predict"
    #     )
    #     metrics = predict_results.metrics
    #     max_predict_samples = len(test_dataset)
    #     metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

    #     trainer.log_metrics("predict", metrics)
    #     trainer.save_metrics("predict", metrics)

    #     # predict on train data
    #     logger.info("*** Predict on training data***")

    #     predict_results2 = trainer.predict(
    #         test_dataset=train_dataset if train_dataset else eval_dataset,
    #         test_examples=seq2seq_train_dataset if seq2seq_train_dataset else seq2seq_eval_dataset,
    #         metric_key_prefix="predict_on_train"
    #     )
    #     metrics2 = predict_results2.metrics
    #     max_predict_samples2 = len(train_dataset)
    #     metrics2["predict_on_train_samples"] = min(max_predict_samples2, len(train_dataset))

    #     trainer.log_metrics("predict_on_train", metrics2)
    #     trainer.save_metrics("predict_on_train", metrics2)

    # # Evaluation for centralized (log for each seperate dev data)
    # if training_args.do_eval:
    #     logger.info("*** Evaluate on eight dev set***")
    #     question_split_eight_eval_name = ['question_split_geoquery_dev.cache', 'question_split_scholar_dev.cache', \
    #                         'question_split_imdb_dev.cache', 'question_split_yelp_dev.cache', \
    #                         'question_split_advising_dev.cache', 'question_split_atis_dev.cache', \
    #                         'question_split_academic_dev.cache', 'question_split_restaurants_dev.cache']

    #     exact_string_match = 0
    #     execution_accuracy_remove_gold_empty = 0
    #     data_total_eval = 0

    #     # for i in range(2):
    #     for i in range(len(question_split_eight_eval_name)):
    #         data_name = question_split_eight_eval_name[i]
    #         data = question_split_eight_eval_dataset[i][0]
            
    #         logger.info("*** Evaluate on " + data_name + "***")
    #         metrics = trainer.evaluate(
    #             eval_dataset=data if data else None,
    #             eval_examples=seq2seq_question_split_eight_eval_dataset[i][0] if seq2seq_question_split_eight_eval_dataset[i][0] else None,
    #             metric_key_prefix="eval" + "_" + data_name
    #         )
            
    #         max_eval_samples = len(data)
    #         metrics["eval_samples" + "_" + data_name] = min(max_eval_samples, len(data))

    #         trainer.log_metrics("eval" + "_" + data_name, metrics)
    #         trainer.save_metrics("eval" + "_" + data_name, metrics)

    #         # calculate for overall eval results
    #         exact_string_match_i = metrics["eval_" + data_name + "_META_TUNING/centralized_with_cell.cfg/exact_string_match"]
    #         execution_accuracy_remove_gold_empty_i = metrics["eval_" + data_name + "_META_TUNING/centralized_with_cell.cfg/execution_accuracy_remove_gold_empty"] 
            
    #         temp_exact_string_match = exact_string_match_i * metrics["eval_samples" + "_" + data_name]
    #         temp_execution_accuracy_remove_gold_empty = execution_accuracy_remove_gold_empty_i * metrics["eval_samples" + "_" + data_name]
            
    #         exact_string_match += temp_exact_string_match
    #         execution_accuracy_remove_gold_empty += temp_execution_accuracy_remove_gold_empty
    #         data_total_eval += metrics["eval_samples" + "_" + data_name]
        
        
    #     logger.info("*** Evaluate on overall eight eval data***")
    #     overall_metrics = {}
    #     overall_metrics["eval_samples_on_overall_eight"] = data_total_eval
    #     overall_metrics["exact_string_match_on_overall_eight"] = exact_string_match/data_total_eval
    #     overall_metrics["execution_accuracy_remove_gold_empty_on_overall_eight"] = execution_accuracy_remove_gold_empty/data_total_eval
    #     trainer.log_metrics("eval_samples_on_overall_eight", overall_metrics)
    #     trainer.save_metrics("eval_samples_on_overall_eight", overall_metrics)


    if training_args.do_predict:
        logger.info("*** Predict on eight test set***")
        question_split_eight_test_name = ['question_split_geoquery_test.cache', 'question_split_scholar_test.cache', \
                            'question_split_imdb_test.cache', 'question_split_yelp_test.cache', \
                            'question_split_advising_test.cache', 'question_split_atis_test.cache', \
                            'question_split_academic_test.cache', 'question_split_restaurants_test.cache']

        # for i in range(2):
        for i in range(len(question_split_eight_test_name)):
            data_name = question_split_eight_test_name[i]
            data = question_split_eight_test_dataset[i][0]
            
            logger.info("*** Predict on test " + data_name + "***")

            predict_results = trainer.predict(
                test_dataset=data if data else question_split_eight_test_dataset[i][0],
                test_examples=seq2seq_question_split_eight_test_dataset[i][0] if seq2seq_question_split_eight_test_dataset[i][0] else seq2seq_question_split_eight_test_dataset[i][0],
                metric_key_prefix="predict" + "_" + data_name
            )
            metrics = predict_results.metrics
            max_predict_samples = len(data)
            metrics["predict_samples" + "_" + data_name] = min(max_predict_samples, len(data))

            trainer.log_metrics("predict" + "_" + data_name, metrics)
            trainer.save_metrics("predict" + "_" + data_name, metrics)


        # predict on dev data
        logger.info("*** Predict on eight dev set***")
        question_split_eight_dev_name = ['question_split_geoquery_dev.cache', 'question_split_scholar_dev.cache', \
                            'question_split_imdb_dev.cache', 'question_split_yelp_dev.cache', \
                            'question_split_advising_dev.cache', 'question_split_atis_dev.cache', \
                            'question_split_academic_dev.cache', 'question_split_restaurants_dev.cache']

        # for i in range(2):
        for i in range(len(question_split_eight_dev_name)):
            data_name = question_split_eight_dev_name[i]
            data = question_split_eight_eval_dataset[i][0]
            
            logger.info("*** Predict on dev " + data_name + "***")

            predict_results = trainer.predict(
                test_dataset=data if data else question_split_eight_eval_dataset[i][0],
                test_examples=seq2seq_question_split_eight_eval_dataset[i][0] if seq2seq_question_split_eight_eval_dataset[i][0] else seq2seq_question_split_eight_test_dataset[i][0],
                metric_key_prefix="predict_on_dev" + "_" + data_name
            )
            metrics = predict_results.metrics
            max_predict_samples = len(data)
            metrics["predict_samples" + "_" + data_name] = min(max_predict_samples, len(data))

            trainer.log_metrics("predict_on_dev" + "_" + data_name, metrics)
            trainer.save_metrics("predict_on_dev" + "_" + data_name, metrics)

        # # predict on train data
        # logger.info("*** Predict on eight train set***")

        # question_split_eight_train_name = ['question_split_geoquery_train.cache', 'question_split_scholar_train.cache', \
        #                     'question_split_imdb_train.cache', 'question_split_yelp_train.cache', \
        #                     'question_split_advising_train.cache', 'question_split_atis_train.cache', \
        #                     'question_split_academic_train.cache', 'question_split_restaurants_train.cache']

        # # for i in range(2):
        # for i in range(len(question_split_eight_train_name)):
        #     data_name = question_split_eight_train_name[i]
        #     data = question_split_eight_train_dataset[i][0]
            
        #     logger.info("*** Predict on train " + data_name + "***")

        #     predict_results2 = trainer.predict(
        #         test_dataset=data if data else question_split_eight_eval_dataset[i][0],
        #         test_examples=seq2seq_question_split_eight_train_dataset[i][0] if seq2seq_question_split_eight_train_dataset[i][0] else seq2seq_question_split_eight_eval_dataset[i][0],
        #         metric_key_prefix="predict_on_train" + "_" + data_name
        #     )
        #     metrics2 = predict_results2.metrics
        #     max_predict_samples2 = len(data)
        #     metrics2["predict_on_train_samples" + "_" + data_name] = min(max_predict_samples2, len(data))

        #     trainer.log_metrics("predict_on_train" + "_" + data_name, metrics2)
        #     trainer.save_metrics("predict_on_train" + "_" + data_name, metrics2)


if __name__ == "__main__":
    main()
