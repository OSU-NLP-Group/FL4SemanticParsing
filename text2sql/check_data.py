import h5py
from tqdm import tqdm
import json

def read_instance_from_h5(data_file, index_list, section, desc):
    train_dataset = []
    test_dataset = []
    if section == "train":
        for i in tqdm(index_list, desc="Loading data from h5 file." + desc):
            example = json.loads(data_file["train/example/" + str(i)][()].decode("utf-8"))
            train_dataset.append(example)
        return train_dataset
    elif section == "test":
        for i in tqdm(index_list, desc="Loading data from h5 file." + desc):
            example = json.loads(data_file["test/example/" + str(i)][()].decode("utf-8"))
            test_dataset.append(example)
        return test_dataset
    else:
        raise ValueError("wrongly input parameters to func read_instance_from_h5")

data_file = h5py.File("/u/tianshuzhang/UnifiedSKG/data/h5_spider/spider_data.h5", "r", swmr=True)
partition_file = h5py.File("/u/tianshuzhang/UnifiedSKG/data/h5_spider/spider_partition_index.h5", "r", swmr=True)
client_index_list = [0,1,2]
train_data_local_dict = {}
test_data_local_dict = {}
train_data_local_num_dict = {}

for client_idx in client_index_list:
            # # TODO: cancel the partiation file usage
         
    train_index_list = partition_file["partition_data"][str(client_idx)]["train"][()]
    test_index_list = partition_file["partition_data"][str(client_idx)]["test"][()]


    train_loader = read_instance_from_h5(
                    data_file, train_index_list, "train", desc=" train data of client_id=%d [_load_federated_data_local] "%client_idx)
    test_loader = read_instance_from_h5(
                    data_file, test_index_list, "test", desc=" test data of client_id=%d [_load_federated_data_local] "%client_idx)
                

    train_data_local_dict[client_idx] = train_loader
    test_data_local_dict[client_idx] = test_loader
    train_data_local_num_dict[client_idx] = len(train_loader)


data_file.close()
partition_file.close()

import pdb
pdb.set_trace()

print(train_data_local_num_dict)
print(train_loader[0],"\n", train_loader[100])
