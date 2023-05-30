import json
import os
import random
import re
import nltk
import h5py
import math

# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))

# from ..FedNLP_spider.data.raw_data_loader.base.base_raw_data_loader import SpiderRawDataLoader

import datasets

import utils.tool
from utils.configue import Configure



class RawDataLoader():
    def __init__(self, data_path):
        # super().__init__(data_path)
        # self.train_file_name = "train-v1.1.json"
        # self.test_file_name = "dev-v1.1.json"
        # self.question_ids = dict()
        self.train_data = self._get_dataset()[0]
        self.test_data = self._get_dataset()[1]  ##actually is dev dataset
        # self.test_data = _get_dataset()[2]

    def load_data(self):
        # TODO: change the first line
        # if len(self.context_X) == 0 or len(self.question_X) == 0 or len(self.Y) == 0:
        #     self.attributes["doc_index"] = dict()
            # train_size = self.process_data_file(os.path.join(self.data_path, self.train_file_name))
            # test_size = self.process_data_file(os.path.join(self.data_path, self.test_file_name))
        train_size = len(self.train_data)
        test_size = len(self.test_data)
        # self.attributes["train_index_list"] = [i for i in range(train_size)]
        # self.attributes["test_index_list"] = [i for i in range(test_size)]
        train_index_list = [i for i in range(train_size)]
        test_index_list = [i for i in range(test_size)]
        # self.attributes["index_list"] = self.attributes["train_index_list"] + self.attributes["test_index_list"]
        return train_index_list, test_index_list

    def process_data_file(self):
        pass
    # def process_data_file(self):
        # cnt = 0
        # with open(file_path, "r", encoding='utf-8') as f:
        #     data = json.load(f)

            # for doc_idx, document in enumerate(data["data"]):
            #     for paragraph in document["paragraphs"]:
            #         for qas in paragraph["qas"]:
            #             for answer in qas["answers"]:
            #                 assert len(self.context_X) == len(self.question_X) == len(self.Y) == len(self.question_ids)
            #                 idx = len(self.context_X)
            #                 self.context_X[idx] = paragraph["context"]
            #                 self.question_X[idx] = qas["question"]
            #                 start = answer["answer_start"]
            #                 end = start + len(answer["text"].rstrip())
            #                 self.Y[idx] = (start, end)
            #                 self.question_ids[idx] = qas["id"]
            #                 self.attributes["doc_index"][idx] = doc_idx
            #                 cnt += 1
        # return cnt
        # data_type is self.train_data or self.test_data
        



        # spider_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        # path=data_args.dataset_paths["spider"], cache_dir=model_args.cache_dir)

    def _get_dataset(self):
        # TODO: from third_party_UnifiedSKG import XX
        # if not self.args.arg_paths:
        #     cache_root = os.path.join('output', 'cache')
        #     os.makedirs(cache_root, exist_ok=True)
        # cache_root = "/u/tianshuzhang/FedNLP_spider/third_party_UnifiedSKG/output/cache"
        # raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(path=self.args.loader_path,
        #                                                                 cache_dir=self.args.data_store_path)
        # seq2seq_dataset_split: tuple = utils.tool.get_constructor("FedNLP_spider.third_party_UnifiedSKG.seq2seq_construction.spider")(self.args).to_seq2seq(
        #     raw_datasets_split, cache_root)
        # else:
        #     cache_root = os.path.join('output', 'cache')
        #     os.makedirs(cache_root, exist_ok=True)
        cache_root = "/u/tianshuzhang/UnifiedSKG/output/cache"
        arg_path = '/u/tianshuzhang/UnifiedSKG/configure/META_TUNING/spider_with_cell.cfg'
        meta_tuning_data = {}
        # for task, arg_path in self.arg_paths:
        args = Configure.Get('/u/tianshuzhang/UnifiedSKG/configure/Salesforce/T5_base_finetune_spider_with_cell_value.cfg')
        task_args = Configure.Get(arg_path)
        task_args.bert = args.bert
        # import pdb
        # pdb.set_trace()
        print('task_args.bert.location:', task_args.bert.location)
        task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(
            path = "/u/tianshuzhang/UnifiedSKG/tasks/spider.py",
            # cache_dir = "/u/tianshuzhang/FedNLP_spider/third_party_UnifiedSKG/data/downloads/extracted/435b5d90f907d194e55ab96066655a1da8f87bf142c3319593f5b75a3c520ce5")
            cache_dir = "/u/tianshuzhang/UnifiedSKG/data")
                    # path=task_args.dataset.loader_path,
                    # cache_dir=task_args.dataset.data_store_path)
        
        task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args).\
                    to_seq2seq(task_raw_datasets_split, cache_root)

        meta_tuning_data[arg_path] = task_seq2seq_dataset_split

        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
                to_seq2seq(meta_tuning_data)

        seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = None, None, None
        if len(seq2seq_dataset_split) == 2:
            seq2seq_train_dataset, seq2seq_eval_dataset = seq2seq_dataset_split
        elif len(seq2seq_dataset_split) == 3:
            seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = seq2seq_dataset_split
        else:
            raise ValueError("Other split not support yet.")

        # # We wrap the "string" seq2seq data into "tokenized tensor".
        # train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
        #                                 seq2seq_train_dataset) if seq2seq_train_dataset else None
        # eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
        #                                 seq2seq_eval_dataset) if seq2seq_eval_dataset else None
        # test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
        #                                 seq2seq_test_dataset) if seq2seq_test_dataset else None
        return seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset

        

    # def generate_h5_file(self, file_path):
    #     f = h5py.File(file_path, "w")
    #     f["attributes"] = json.dumps(self.attributes)
    #     for key in self.query.keys():
    #         f["context_X/" + str(key)] = self.context_X[key]
    #         f["question_X/" + str(key)] = self.question_X[key]
    #         f["Y/" + str(key)] = self.Y[key]
    #         f["question_ids/" + str(key)] = self.question_ids[key]
    #     f.close()

    def generate_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        # f["attributes"] = json.dumps(self.attributes)
        for data_idx in range(len(self.train_data)):
            f["train/query/" + str(data_idx)] = self.train_data[data_idx]["query"]
            f["train/question/" + str(data_idx)] = self.train_data[data_idx]["question"]
            f["train/db_id/" + str(data_idx)] = self.train_data[data_idx]["db_id"]
            # f["train/db_path/" + str(data_idx)] = self.train_data[data_idx]["db_path"]  ## may need to change
            f["train/db_path/" + str(data_idx)] = "/u/tianshuzhang/UnifiedSKG/data/downloads/extracted/435b5d90f907d194e55ab96066655a1da8f87bf142c3319593f5b75a3c520ce5/spider/database"
            f["train/db_table_names/" + str(data_idx)] = self.train_data[data_idx]["db_table_names"]
            f["train/db_column_names/" + str(data_idx)] = json.dumps(self.train_data[data_idx]["db_column_names"])
            f["train/db_column_types" + str(data_idx)] = self.train_data[data_idx]["db_column_types"]
            f["train/db_primary_keys/" + str(data_idx)] = json.dumps(self.train_data[data_idx]["db_primary_keys"])
            f["train/db_foreign_keys/" + str(data_idx)] = json.dumps(self.train_data[data_idx]["db_foreign_keys"])
            f["train/serialized_schema/" + str(data_idx)] = self.train_data[data_idx]["serialized_schema"]
            f["train/struct_in/" + str(data_idx)] = self.train_data[data_idx]["struct_in"]
            f["train/text_in/" + str(data_idx)] = self.train_data[data_idx]["text_in"]
            f["train/seq_out/" + str(data_idx)] = self.train_data[data_idx]["seq_out"]
            f["train/section/" + str(data_idx)] = self.train_data[data_idx]["section"]
            f["train/arg_path/" + str(data_idx)] = self.train_data[data_idx]["arg_path"]
            f["train/example/" + str(data_idx)] = json.dumps(self.train_data[data_idx])

        for data_idx in range(len(self.test_data)):
            f["test/query/" + str(data_idx)] = self.test_data[data_idx]["query"]
            f["test/question/" + str(data_idx)] = self.test_data[data_idx]["question"]
            f["test/db_id/" + str(data_idx)] = self.test_data[data_idx]["db_id"]
            # f["test/db_path/" + str(data_idx)] = self.test_data[data_idx]["db_path"] ## may need to change
            f["test/db_path/" + str(data_idx)] = "/u/tianshuzhang/UnifiedSKG/data/downloads/extracted/435b5d90f907d194e55ab96066655a1da8f87bf142c3319593f5b75a3c520ce5/spider/database"
            f["test/db_table_names/" + str(data_idx)] = self.test_data[data_idx]["db_table_names"]
            f["test/db_column_names/" + str(data_idx)] = json.dumps(self.test_data[data_idx]["db_column_names"])
            f["test/db_column_types" + str(data_idx)] = self.test_data[data_idx]["db_column_types"]
            f["test/db_primary_keys/" + str(data_idx)] = json.dumps(self.test_data[data_idx]["db_primary_keys"])
            f["test/db_foreign_keys/" + str(data_idx)] = json.dumps(self.test_data[data_idx]["db_foreign_keys"])
            f["test/serialized_schema/" + str(data_idx)] = self.test_data[data_idx]["serialized_schema"]
            f["test/struct_in/" + str(data_idx)] = self.test_data[data_idx]["struct_in"]
            f["test/text_in/" + str(data_idx)] = self.test_data[data_idx]["text_in"]
            f["test/seq_out/" + str(data_idx)] = self.test_data[data_idx]["seq_out"]
            f["test/section/" + str(data_idx)] = self.test_data[data_idx]["section"]
            f["test/arg_path/" + str(data_idx)] = self.test_data[data_idx]["arg_path"]
            f["test/example/" + str(data_idx)] = json.dumps(self.test_data[data_idx])

        f.close()
    

    def spider_uniform_partition(self, file_path, n_clients, train_index_list, test_index_list=None):
        f = h5py.File(file_path, "w")

        partition_dict = dict()
        partition_dict["n_clients"] = n_clients
        partition_dict["partition_data"] = dict()
        train_index_list = train_index_list.copy()
        random.shuffle(train_index_list)
        train_batch_size = math.ceil(len(train_index_list) / n_clients)

        test_batch_size = None

        if test_index_list is not None:
            test_index_list = test_index_list.copy()
            random.shuffle(test_index_list)
            test_batch_size = math.ceil(len(test_index_list) / n_clients)
        for i in range(n_clients):
            train_start = i * train_batch_size
            partition_dict["partition_data"][i] = dict()
            train_set = train_index_list[train_start: train_start + train_batch_size]
            if test_index_list is None:
                random.shuffle(train_set)
                train_num = int(len(train_set) * 0.8)
                partition_dict["partition_data"][i]["train"] = train_set[:train_num]
                partition_dict["partition_data"][i]["test"] = train_set[train_num:]
                f["partition_data/" + str(i) + "/train"] = partition_dict["partition_data"][i]["train"]
                f["partition_data/" + str(i) + "/test"] = partition_dict["partition_data"][i]["test"]
            else:
                test_start = i * test_batch_size
                partition_dict["partition_data"][i]["train"] = train_set
                partition_dict["partition_data"][i]["test"] = test_index_list[test_start:test_start + test_batch_size]
                f["partition_data/" + str(i) + "/train"] = partition_dict["partition_data"][i]["train"]
                f["partition_data/" + str(i) + "/test"] = partition_dict["partition_data"][i]["test"]
        
        f.close()
        return partition_dict




def main():
    spider_data_loader = RawDataLoader("")
    print(spider_data_loader.test_data[0].keys())
    spider_data_loader.generate_h5_file("./data/h5_spider/spider_data.h5")
    train_index_list, test_index_list = spider_data_loader.load_data()
    partition_dict = spider_data_loader.spider_uniform_partition("./data/h5_spider/spider_partition_index.h5", 3, train_index_list, test_index_list = test_index_list)


if __name__ == "__main__":
    main()