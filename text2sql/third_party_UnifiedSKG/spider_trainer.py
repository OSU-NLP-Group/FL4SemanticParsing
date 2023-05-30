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

# Huggingface realized the "Seq2seqTrainingArguments" which is the same with "WrappedSeq2SeqTrainingArguments"
# in transformers==4.10.1 during our work.
logger = logging.getLogger(__name__)


class SpiderTrainer(ClientTrainer):
    def __init__(self, args, training_args, model, evaluator, model_tokenizer, seq2seq_train_dataset=None, seq2seq_eval_dataset=None, seq2seq_test_dataset=None):
        self.args = args
        self.training_args = training_args
        self.model = model
        self.evaluator = evaluator
        self.model_tokenizer = model_tokenizer
        self.set_data(seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset)
        self.loss_signal = None
        self.relative_loss_signal = None
        self.iteration_step = None

    # def get_model_params(self):
    #     return self.model.cpu().state_dict()
    
    def get_model_params(self):
        # model = self.model.cpu().state_dict()
        # loss_signal = self.loss_signal
        # return (model, loss_signal)
        model = self.model.cpu().state_dict()
        loss_signal = self.loss_signal
        relative_loss_signal = self.relative_loss_signal
        iteration_step = self.iteration_step
        return (model, loss_signal, relative_loss_signal, iteration_step)

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
        # print(f'round_idx = {self.args.round_idx}')
        # logging.info("----------round_idx = {}--------".format(self.args.round_idx))
        # import pdb
        # pdb.set_trace()

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

        os.makedirs(training_args.output_dir, exist_ok=True)

        # print("self.args.fl_algorithm:", self.args.fl_algorithm)
        # print("self.args.fedprox_mu:", self.args.fedprox_mu)
        # import pdb
        # pdb.set_trace()
        # Initialize our Trainer
        # print("self.args.fl_algorithm:", self.args.fl_algorithm)
        # print("train_dataset[0].keys():", train_dataset[0].keys())
        # print("self.seq2seq_train_dataset[0].keys():", self.seq2seq_train_dataset[0].keys())
        # print("self.seq2seq_train_dataset[0]['db_id']:", self.seq2seq_train_dataset[0]["db_id"])
        # import pdb
        # pdb.set_trace()
        if self.args.fl_algorithm in ["FedOpt", "FedAvg"] and self.seq2seq_train_dataset[0]["db_id"] == "atis":
            training_args.per_device_train_batch_size = 8 #also change all output_length, so this would vary
        if self.args.fl_algorithm == "FedProx" and self.seq2seq_train_dataset[0]["db_id"] == "atis":
            training_args.per_device_train_batch_size = 8 #also change all output_length, so this would vary
        if self.seq2seq_train_dataset[0]["db_id"] in ["restaurants", "academic", "imdb", "yelp"]:
            training_args.learning_rate = 1e-3
        if self.seq2seq_train_dataset[0]["db_id"] in ["atis", "advising"]:
            training_args.num_train_epochs = 6
        
        # epoch_dict = {'geoquery': [6, 3, 9, 10, 10, 5, 7, 8, 10, 6, 8, 6, 4, 6, 9, 5, 4, 3, 8, 4, 4, 9, 7, 7, 6, 10, 7, 7, 9, 9, 4, 3, 4, 7, 5, 6, 10, 5, 8, 8, 5, 6, 9, 10, 3, 9, 7, 9, 10, 4, 7, 6, 7, 7, 5, 6], 'advising': [3, 4, 3, 4, 6, 5, 4, 4, 5, 5, 3, 3, 4, 5, 6, 3, 6, 6, 3, 4, 6, 5, 6, 3, 4, 3, 6, 5, 3, 5, 5, 5, 3, 3, 4, 6, 3, 4, 5, 6, 3, 4, 5, 5, 3, 3, 3, 5, 4, 3, 6, 6, 6, 6, 5, 4], 'atis': [6, 5, 5, 6, 4, 5, 5, 6, 5, 5, 3, 3, 3, 5, 4, 6, 4, 6, 3, 4, 4, 5, 6, 5, 5, 5, 6, 5, 3, 4, 3, 3, 3, 6, 5, 3, 4, 5, 3, 4, 6, 3, 4, 5, 4, 3, 5, 5, 4, 5, 5, 5, 5, 5, 4, 5], 'scholar': [9, 4, 4, 9, 8, 10, 5, 4, 9, 8, 5, 10, 4, 4, 10, 3, 10, 6, 9, 9, 5, 4, 7, 3, 7, 7, 7, 7, 3, 7, 3, 5, 6, 4, 10, 8, 4, 4, 9, 7, 7, 10, 5, 10, 8, 6, 6, 8, 3, 8, 7, 10, 9, 9, 7, 8], 'yelp': [6, 4, 10, 7, 7, 4, 9, 8, 9, 6, 8, 5, 9, 4, 10, 5, 10, 5, 4, 9, 6, 8, 4, 10, 3, 5, 4, 3, 4, 5, 9, 6, 10, 9, 3, 6, 8, 3, 6, 7, 6, 10, 4, 7, 7, 5, 6, 10, 4, 9, 6, 9, 10, 9, 8, 9], 'imdb': [9, 10, 9, 4, 4, 4, 4, 10, 10, 4, 10, 8, 10, 7, 9, 10, 4, 10, 9, 3, 6, 4, 6, 10, 6, 3, 3, 6, 8, 7, 8, 8, 5, 5, 5, 10, 4, 4, 4, 9, 6, 6, 10, 6, 10, 5, 7, 8, 9, 3, 6, 4, 5, 6, 4, 3], 'restaurants': [8, 5, 10, 6, 7, 4, 8, 8, 10, 4, 7, 3, 6, 9, 7, 9, 6, 9, 6, 9, 3, 3, 10, 7, 8, 7, 7, 9, 6, 7, 5, 9, 5, 10, 8, 10, 5, 10, 10, 6, 3, 8, 7, 3, 5, 10, 5, 4, 3, 4, 8, 8, 10, 3, 4, 7], 'academic': [3, 9, 9, 9, 8, 3, 8, 10, 9, 8, 9, 6, 10, 4, 5, 6, 9, 7, 10, 10, 4, 4, 4, 6, 3, 8, 4, 10, 10, 8, 9, 4, 3, 6, 10, 6, 6, 8, 6, 5, 5, 7, 9, 7, 6, 9, 6, 4, 3, 4, 9, 4, 5, 6, 5, 6]}
        # training_args.num_train_epochs = epoch_dict[self.seq2seq_train_dataset[0]["db_id"]][0]
        # del epoch_dict[self.seq2seq_train_dataset[0]["db_id"]][0]
        # print("data, len(epoch_list:",self.seq2seq_train_dataset[0]["db_id"], len(epoch_dict[self.seq2seq_train_dataset[0]["db_id"]]) )

        # import random

        # # print(self.seq2seq_train_dataset[0]["db_id"], training_args.num_train_epochs)
        # if self.seq2seq_train_dataset[0]["db_id"] == "atis":
        #     random.seed()
        #     training_args.num_train_epochs = random.choice([3,4,5,6])
        #     with open("epoch_trace.txt", "a+") as writers: 
        #         writers.write(self.seq2seq_train_dataset[0]["db_id"] + ',' + str(training_args.num_train_epochs) + "\n")

        # elif self.seq2seq_train_dataset[0]["db_id"] == "advising":
        #     random.seed()
        #     training_args.num_train_epochs = random.choice([3,4,5,6])
        #     with open("epoch_trace.txt", "a+") as writers: 
        #         writers.write(self.seq2seq_train_dataset[0]["db_id"] + ',' + str(training_args.num_train_epochs) + "\n")

        # elif self.seq2seq_train_dataset[0]["db_id"] == "geoquery":
        #     random.seed()
        #     training_args.num_train_epochs = random.choice([3,4,5,6,7,8,9,10])
        #     with open("epoch_trace.txt", "a+") as writers: 
        #         writers.write(self.seq2seq_train_dataset[0]["db_id"] + ',' + str(training_args.num_train_epochs) + "\n")

        # elif self.seq2seq_train_dataset[0]["db_id"] == "scholar":
        #     random.seed()
        #     training_args.num_train_epochs = random.choice([3,4,5,6,7,8,9,10])
        #     with open("epoch_trace.txt", "a+") as writers: 
        #         writers.write(self.seq2seq_train_dataset[0]["db_id"] + ',' + str(training_args.num_train_epochs) + "\n")

        # elif self.seq2seq_train_dataset[0]["db_id"] == "restaurants":
        #     random.seed()
        #     training_args.num_train_epochs = random.choice([3,4,5,6,7,8,9,10])
        #     with open("epoch_trace.txt", "a+") as writers: 
        #         writers.write(self.seq2seq_train_dataset[0]["db_id"] + ',' + str(training_args.num_train_epochs) + "\n")

        # elif self.seq2seq_train_dataset[0]["db_id"] == "academic":
        #     random.seed()
        #     training_args.num_train_epochs = random.choice([3,4,5,6,7,8,9,10])
        #     with open("epoch_trace.txt", "a+") as writers: 
        #         writers.write(self.seq2seq_train_dataset[0]["db_id"] + ',' + str(training_args.num_train_epochs) + "\n")

        # elif self.seq2seq_train_dataset[0]["db_id"] == "imdb":
        #     random.seed()
        #     training_args.num_train_epochs = random.choice([3,4,5,6,7,8,9,10])
        #     with open("epoch_trace.txt", "a+") as writers: 
        #         writers.write(self.seq2seq_train_dataset[0]["db_id"] + ',' + str(training_args.num_train_epochs) + "\n")

        # else:
        #     assert self.seq2seq_train_dataset[0]["db_id"] == "yelp"
        #     random.seed()
        #     training_args.num_train_epochs = random.choice([3,4,5,6,7,8,9,10])
        #     with open("epoch_trace.txt", "a+") as writers: 
        #         writers.write(self.seq2seq_train_dataset[0]["db_id"] + ',' + str(training_args.num_train_epochs) + "\n")
        # # print(self.seq2seq_train_dataset[0]["db_id"], training_args.num_train_epochs)

        # random.seed(0)


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
        print(f'round_idx = {round_idx}')
        print(f'Trainer on device {device} build successfully.')

        # # Load model weights (for --do_train=False or post finetuning).
        # if training_args.load_weights_from:
        #     state_dict = torch.load(os.path.join(training_args.load_weights_from, transformers.WEIGHTS_NAME), map_location="cpu")
        #     trainer.model.load_state_dict(state_dict, strict=True)
        #     # release memory
        #     del state_dict

        # if args.load_multiple_prefix_module_weights_from:
        #     reconstruct_state_dict = OrderedDict()

        #     # load prefix modules
        #     for task_name, module_weight_location in args.load_multiple_prefix_module_weights_from:
        #         state_dict = torch.load(os.path.join(module_weight_location, transformers.WEIGHTS_NAME), map_location="cpu")
        #         MULTI_PREFIX_ATTR_NAME = "multi_prefix"
        #         for weight_name, stored_tensor in state_dict.items():
        #             if str(weight_name).startswith("pretrain_model"):
        #                 continue  # skip the pretrained model and we will load a new one from another place
        #             reconstruct_state_dict['{}.{}.{}'.format(MULTI_PREFIX_ATTR_NAME, "_".join(task_name.split("_")[:-1]), weight_name)] = stored_tensor
        #             # extract the prefix part and add them to dict

        #     # give it into the model
        #     trainer.model.load_state_dict(reconstruct_state_dict, strict=False)

        #     # release memory
        #     del reconstruct_state_dict

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint

            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            # trainer.save_model()  # Saves the tokenizer too for easy upload

            self.loss_signal = trainer.loss_signal
            self.relative_loss_signal = trainer.relative_loss_signal
            self.iteration_step = trainer.iteration_step
            print(self.loss_signal, self.relative_loss_signal, trainer.loss_signal, do_train, do_eval, self.iteration_step)
            # import pdb
            # pdb.set_trace()

            metrics = train_result.metrics
            max_train_samples = len(train_dataset)
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            # trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate(
                metric_key_prefix="eval"
            )
            max_eval_samples = len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

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
