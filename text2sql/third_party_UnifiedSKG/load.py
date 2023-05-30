import datasets

task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(
            path = "/u/tianshuzhang/UnifiedSKG/tasks/spider.py",
            # cache_dir = "/u/tianshuzhang/FedNLP_spider/third_party_UnifiedSKG/data/downloads/extracted/435b5d90f907d194e55ab96066655a1da8f87bf142c3319593f5b75a3c520ce5")
            cache_dir = "/u/tianshuzhang/UnifiedSKG/data")
                    # path=task_args.dataset.loader_path,
          
print(task_raw_datasets_split)