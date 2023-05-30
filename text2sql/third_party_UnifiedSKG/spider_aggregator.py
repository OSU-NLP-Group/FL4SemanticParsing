import logging
import os
import time

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
import third_party_UnifiedSKG.utils.tool
from third_party_UnifiedSKG.utils.configue import Configure
from third_party_UnifiedSKG.utils.dataset import TokenizedDataset
from third_party_UnifiedSKG.utils.trainer import EvaluateFriendlySeq2SeqTrainerSetGPU
from third_party_UnifiedSKG.utils.training_arguments import WrappedSeq2SeqTrainingArguments
from fedml.core import ClientTrainer


import wandb
import os
from multiprocessing import Pool
import torch
from tqdm import tqdm
from fedml.core import ServerAggregator

# Huggingface realized the "Seq2seqTrainingArguments" which is the same with "WrappedSeq2SeqTrainingArguments"
# in transformers==4.10.1 during our work.
logger = logging.getLogger(__name__)

class SpiderAggregator(ServerAggregator):
    def __init__(self, args, training_args, model, evaluator, model_tokenizer, seq2seq_train_dataset=None, seq2seq_eval_dataset=None, seq2seq_test_dataset=None):
        self.args = args
        self.training_args = training_args
        self.model = model
        self.evaluator = evaluator
        self.model_tokenizer = model_tokenizer
        self.set_data(seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def set_data(self, seq2seq_train_dataset=None, seq2seq_eval_dataset=None, seq2seq_test_dataset=None):
        self.seq2seq_train_dataset = seq2seq_train_dataset
        self.seq2seq_eval_dataset = seq2seq_eval_dataset
        self.seq2seq_test_dataset = seq2seq_test_dataset

        self.train_dl = TokenizedDataset(self.args, self.training_args, self.model_tokenizer,
                                     seq2seq_train_dataset) if seq2seq_train_dataset else None
        self.eval_dl = TokenizedDataset(self.args, self.training_args, self.model_tokenizer,
                                    seq2seq_eval_dataset) if seq2seq_eval_dataset else None
        self.test_dl = TokenizedDataset(self.args, self.training_args, self.model_tokenizer,
                                    seq2seq_test_dataset) if seq2seq_test_dataset else None

    def train(self, train_data, device, args, test_data=None):
        logging.info("train_model self.device: " + str(device))
        self.set_data(seq2seq_train_dataset=train_data, seq2seq_eval_dataset=test_data)
        ## do_eval using True or False? maybe False since we don't select best model for client?
        self.train_eval_predict(device = device, do_train = True, do_eval = False, do_predict = False)
    
    def test(self, test_data, device, round_idx, args):
        logging.info("eval_model self.device: " + str(device))
        self.set_data(seq2seq_eval_dataset=test_data)
        self.train_eval_predict(device = device, do_train = False, do_eval = True, do_predict = False, round_idx = round_idx)

    # ### test client X data on the server  
    # def test_on_the_server(
    #     self, train_data_local_dict, test_data_local_dict, device, args=None
    # ) -> bool:
    #     logging.info("----------test_on_the_server--------")
    #     # f1_list, metric_list = [], []
    #     for client_idx in test_data_local_dict.keys():
    #         test_data = test_data_local_dict[client_idx]
    #         self.test(test_data, device, args)
    #         # metric_list.append(metrics)
    #         # f1_list.append(metrics["rouge_score"])
    #     #     logging.info(
    #     #         "Client {}, Test rouge_score = {}".format(
    #     #             client_idx, metrics["rouge_score"]
    #     #         )
    #     #     )
    #     # avg_accuracy = np.mean(np.array(f1_list))
    #     # logging.info("Test avg rouge_score = {}".format(avg_accuracy))
    #     logging.info("Client {} data is testing on the server".format(client_idx))
    #     return True

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, round_idx, args=None
    ) -> bool:
        logging.info("----------test_on_the_server--------")
        print(f'round_idx = {round_idx}')
        # f1_list, metric_list = [], []
        global_test_data = []
        for client_idx in test_data_local_dict.keys():
            global_test_data.extend(test_data_local_dict[client_idx])
        self.test(global_test_data, device, round_idx, args)
        global_test_data_size = len(global_test_data)
            # metric_list.append(metrics)
            # f1_list.append(metrics["rouge_score"])
        #     logging.info(
        #         "Client {}, Test rouge_score = {}".format(
        #             client_idx, metrics["rouge_score"]
        #         )
        #     )
        # avg_accuracy = np.mean(np.array(f1_list))
        # logging.info("Test avg rouge_score = {}".format(avg_accuracy))
        # logging.info("Client {} data is testing on the server".format(client_idx))
        logging.info("{} test data is testing on the server".format(global_test_data_size))

        return True

    def train_eval_predict(self, device=None, do_train = False, do_eval = False, do_predict = False, round_idx = None):
        args = self.args
        training_args = self.training_args
        model = self.model
        evaluator = self.evaluator
        model_tokenizer = self.model_tokenizer
        train_dataset = self.train_dl
        eval_dataset = self.eval_dl
        test_dataset = self.test_dl

        # self.model.to(device)
        # training_args.device = device

        # print(training_args)
        # print(1)
        # import pdb
        # pdb.set_trace()
        
        training_args.do_train = do_train
        training_args.do_eval = do_eval
        training_args.do_predict = do_predict

        seq2seq_train_dataset = self.seq2seq_train_dataset
        seq2seq_eval_dataset = self.seq2seq_eval_dataset
        seq2seq_test_dataset = self.seq2seq_test_dataset

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
        import re
        temp_output_dir = re.sub(r'\\round*',"", training_args.output_dir)
        training_args.output_dir = temp_output_dir + '\\round' + str(round_idx)
        os.makedirs(training_args.output_dir, exist_ok=True)

        # import pdb
        # pdb.set_trace()
        # if self.args.fl_algorithm in ["FedOpt", "FedAvg"] and self.seq2seq_train_dataset[0]["db_id"] == "atis":
        #     training_args.per_device_train_batch_size = 8 #also change all output_length, so this would vary
        # if self.args.fl_algorithm == "FedProx" and self.seq2seq_train_dataset[0]["db_id"] == "atis":
        #     training_args.per_device_train_batch_size = 8 #also change all output_length, so this would vary
        # if self.seq2seq_train_dataset[0]["db_id"] in ["restaurants", "academic", "imdb", "yelp"]:
        #     training_args.learning_rate = 1e-3

        # Initialize our Trainer
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.seq2seq.patience if args.seq2seq.patience else 5)
        trainer = EvaluateFriendlySeq2SeqTrainerSetGPU(
            fl_algorithm = self.args.fl_algorithm,
            fedprox_mu = self.args.fedprox_mu,
            device = device,
            args=training_args,
            model=model,
            evaluator=evaluator,
            # We name it "evaluator" while the hugging face call it "Metric",
            # they are all f(predictions: List, references: List of dict) = eval_result: dict
            tokenizer=model_tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_examples=seq2seq_eval_dataset,
            ## Fix me: uncomment this later
            # wandb_run_dir=wandb.run.dir if "wandb" in training_args.report_to and training_args.local_rank <= 0 else None,
            callbacks=[early_stopping_callback],
        )
        print(f'Trainer on device {device} build successfully.')

        # # Load model weights (for --do_train=False or post finetuning).
        # if training_args.load_weights_from:
        #     state_dict = torch.load(os.path.join(training_args.load_weights_from, transformers.WEIGHTS_NAME), map_location="cpu")
        #     trainer.model.load_state_dict(state_dict, strict=True)
        #     # release memory
        #     del state_dict

        # if args.load_multiple_prefix_module_weights_from:
        #     reconstruct_state_dict = OrderedDict()

            # # load prefix modules
            # for task_name, module_weight_location in args.load_multiple_prefix_module_weights_from:
            #     state_dict = torch.load(os.path.join(module_weight_location, transformers.WEIGHTS_NAME), map_location="cpu")
            #     MULTI_PREFIX_ATTR_NAME = "multi_prefix"
            #     for weight_name, stored_tensor in state_dict.items():
            #         if str(weight_name).startswith("pretrain_model"):
            #             continue  # skip the pretrained model and we will load a new one from another place
            #         reconstruct_state_dict['{}.{}.{}'.format(MULTI_PREFIX_ATTR_NAME, "_".join(task_name.split("_")[:-1]), weight_name)] = stored_tensor
            #         # extract the prefix part and add them to dict

            # # give it into the model
            # trainer.model.load_state_dict(reconstruct_state_dict, strict=False)

            # # release memory
            # del reconstruct_state_dict

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

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            trainer.save_model()

            metrics = trainer.evaluate(
                metric_key_prefix="eval"
            )
            max_eval_samples = len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            trainer.save_state()

        if training_args.do_predict:
            logger.info("*** Predict ***")

            predict_results = trainer.predict(
                test_dataset=test_dataset if test_dataset else eval_dataset,
                test_examples=seq2seq_test_dataset if seq2seq_test_dataset else seq2seq_eval_dataset,
                metric_key_prefix="predict"
            )
            metrics = predict_results.metrics
            max_predict_samples = len(test_dataset)
            metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)


# if __name__ == "__main__":
#     main()
