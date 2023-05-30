import logging
import torch
import transformers

import fedml
from data.data_loader import load
from fedml import FedMLRunner
from fedml.model.nlp.model_args import *
# from trainer.seq2seq_trainer import MyModelTrainer as MySSTrainer
from third_party_UnifiedSKG.spider_trainer import SpiderTrainer
from third_party_UnifiedSKG.spider_aggregator import SpiderAggregator
import third_party_UnifiedSKG.utils.tool
from third_party_UnifiedSKG.utils.configue import Configure

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6,7'   # strawberry2

# Huggingface realized the "Seq2seqTrainingArguments" which is the same with "WrappedSeq2SeqTrainingArguments"
# in transformers==4.10.1 during our work.
logger = logging.getLogger(__name__)


def create_model(args, device, training_args, output_dim=1):
    model_name = args.model
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
    print("reached_here: in torch_main create_model function...")
    # trainer = MySSTrainer(model_args, device, model, tokenizer=tokenizer)

    backup_model_args = Configure.Get(training_args.cfg)
    backup_model_args.fl_algorithm = args.federated_optimizer
    backup_model_args.fedprox_mu = args.fedprox_mu
    
    evaluator = third_party_UnifiedSKG.utils.tool.get_evaluator(backup_model_args.evaluate.tool)(backup_model_args)
    client_model = third_party_UnifiedSKG.utils.tool.get_model(backup_model_args.model.name)(backup_model_args)
    model_tokenizer = client_model.tokenizer

    if training_args.load_weights_from:
        print("loading weights from checkpoints...")
        state_dict = torch.load(os.path.join(training_args.load_weights_from, transformers.WEIGHTS_NAME), map_location="cpu")
        client_model.load_state_dict(state_dict, strict=True)
        # release memory
        del state_dict

    # client_trainer = SpiderTrainer(backup_model_args, training_args, model, evaluator, model_tokenizer, None, None, None)
    client_trainer = SpiderTrainer(backup_model_args, training_args, client_model, evaluator, model_tokenizer, None, None, None)

    return client_model, client_trainer, backup_model_args, training_args, evaluator, model_tokenizer


if __name__ == "__main__":

    
    from transformers import HfArgumentParser
    from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
    from third_party_UnifiedSKG.utils.training_arguments import WrappedSeq2SeqTrainingArguments
    from fl_arguments import FLArguments
    
    parser = HfArgumentParser(
        (WrappedSeq2SeqTrainingArguments,)
    )
    # parser = HfArgumentParser(
    #     (FLArguments, WrappedSeq2SeqTrainingArguments)
    # )
    # fl_args: FLArguments
    training_args: Seq2SeqTrainingArguments
    if len(sys.argv) == 4 and sys.argv[3].endswith(".json"):
        # print(1)
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        training_args, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[3])
        )
  
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load(args)

    client_model, client_trainer, backup_model_args, training_args, evaluator, model_tokenizer = create_model(args, output_dim, training_args)
  
    #add for 0.7.327 fedml version
    aggregator = SpiderAggregator(backup_model_args, training_args, client_model, evaluator, model_tokenizer, None, None, None)
    # start training
    # fedml_runner = FedMLRunner(args, device, dataset, model, trainer)
    
    # change for 0.7.327 fedml version
    fedml_runner = FedMLRunner(args, device, dataset, client_model, client_trainer, aggregator)
    fedml_runner.run()
