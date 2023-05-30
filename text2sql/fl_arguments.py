from dataclasses import dataclass, field

@dataclass
class FLArguments:
    """
    Arguments used in FedNLP repo.
    """

    run_id: int = field(
        default=0,
        metadata={"help": ""},
    )
    is_debug_mode: int = field(
        default=0,
        metadata={"help": ""},
    )

    dataset: str = field(
        default="",
        metadata={"help": ""},
    )

    data_file_path: str = field(
        default="",
        metadata={"help": ""},
    )

    partition_file_path: str = field(
        default="",
        metadata={"help": ""},
    )

    partition_method: str = field(
        default="",
        metadata={"help": ""},
    )

    model_type: str = field(
        default="",
        metadata={"help": ""},
    )
    model_class: str = field(
        default="",
        metadata={"help": ""},
    )
    model: str = field(
        default="",
        metadata={"help": ""},
    )

    do_lower_case: str  = field(
        default="",
        metadata={"help": ""},
    )

    batch_size: int = field(
        default=8,
        metadata={"help": ""},
    )
    eval_batch_size: int = field(
        default=128,
        metadata={"help": ""},
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": ""},
    )
    n_gpu: int = field(
        default=1,
        metadata={"help": ""},
    )
    fp16: bool = field(
        default=False,
        metadata={"help": ""},
    )
    random_seed: int = field(
        default=42,
        metadata={"help": ""},
    )
    output_dir: str = field(
        default="",
        metadata={"help": ""},
    )
    federated_optimizer: str  = field(
        default="FedAvg",
        metadata={"help": ""},
    )
    backend: str = field(
        default="MPI",
        metadata={"help": ""},
    )
    comm_round: int = field(
        default=10,
        metadata={"help": ""},
    )
    is_mobile: int = field(
        default=1,
        metadata={"help": ""},
    )
    client_num_in_total: int  = field(
        default=1,
        metadata={"help": ""},
    )
    client_num_per_round: int = field(
        default=4,
        metadata={"help": ""},
    )
    epochs: int = field(
        default=3,
        metadata={"help": ""},
    )
    gradient_accumulation_steps: int  = field(
        default=1,
        metadata={"help": ""},
    )
    client_optimizer: str = field(
        default="adam",
        metadata={"help": ""},
    )

    learning_rate: float = field(
        default=0.1,
        metadata={"help": ""},
    )
    weight_decay: float = field(
        default=0,
        metadata={"help": ""},
    )
    clip_grad_norm: int = field(
        default=0,
        metadata={"help": ""},
    )
    server_optimizer: str = field(
        default="sgd",
        metadata={"help": ""},
    )
    server_lr: float = field(
        default=0.1,
        metadata={"help": ""},
    )
    server_momentum: float = field(
        default=0,
        metadata={"help": ""},
    )
    fedprox_mu: float = field(
        default=1,
        metadata={"help": ""},
    )
    evaluate_during_training: bool = field(
        default=False,
        metadata={"help": ""},
    )
    evaluate_during_training_steps: int = field(
        default=100,
        metadata={"help": ""},
    )
    frequency_of_the_test: int = field(
        default=1,
        metadata={"help": ""},
    )
    gpu_mapping_file: str  = field(
        default="gpu_mapping.yaml",
        metadata={"help": ""},
    )
    gpu_mapping_key: str = field(
        default="mapping_default",
        metadata={"help": ""},
    )
    ci: int = field(
        default=0,
        metadata={"help": ""},
    )
    reprocess_input_data: str = field(
        default="",
        metadata={"help": ""},
    )
    freeze_layers: str = field(
        default="",
        metadata={"help": ""},
    )
    #     :  = field(
    #     default=,
    #     metadata={"help": ""},
    # )


    # parser.add_argument("--is_debug_mode", default=0, type=int, help="is_debug_mode")

    # # Data related
    # # TODO: list all dataset names:
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="squad_1.1",
    #     metavar="N",
    #     help="dataset used for training",
    # )

    # parser.add_argument(
    #     "--data_file_path",
    #     type=str,
    #     default="/home/ubuntu/fednlp_data/data_files/cornell_movie_dialogue_data.h5",
    #     help="data h5 file path",
    # )

    # parser.add_argument(
    #     "--partition_file_path",
    #     type=str,
    #     default="/home/ubuntu/fednlp_data//partition_files/cornell_movie_dialogue_partition.h5",
    #     help="partition h5 file path",
    # )

    # parser.add_argument(
    #     "--partition_method", type=str, default="uniform", help="partition method"
    # )

    # # Model related
    # parser.add_argument(
    #     "--model_type",
    #     type=str,
    #     default="bart",
    #     metavar="N",
    #     help="transformer model type",
    # )

    # parser.add_argument(
    #     "--model_class",
    #     type=str,
    #     default="transformer",
    #     metavar="N",
    #     help="model class",
    # )

    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     default="facebook/bart-base",
    #     metavar="N",
    #     help="transformer model name",
    # )
    # parser.add_argument(
    #     "--do_lower_case",
    #     type=bool,
    #     default=True,
    #     metavar="N",
    #     help="transformer model name",
    # )

    # # Learning related
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=8,
    #     metavar="N",
    #     help="input batch size for training (default: 8)",
    # )
    # parser.add_argument(
    #     "--eval_batch_size",
    #     type=int,
    #     default=8,
    #     metavar="N",
    #     help="input batch size for evaluation (default: 8)",
    # )

    # parser.add_argument(
    #     "--max_seq_length",
    #     type=int,
    #     default=128,
    #     metavar="N",
    #     help="maximum sequence length (default: 128)",
    # )

    # parser.add_argument(
    #     "--n_gpu", type=int, default=1, metavar="EP", help="how many gpus will be used "
    # )

    # parser.add_argument(
    #     "--fp16", default=False, action="store_true", help="if enable fp16 for training"
    # )
    # parser.add_argument(
    #     "--random_seed", type=int, default=42, metavar="N", help="random seed"
    # )

    # # IO related
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="/tmp/",
    #     metavar="N",
    #     help="path to save the trained results and ckpts",
    # )

    # # Federated Learning related
    # parser.add_argument(
    #     "--federated_optimizer",
    #     type=str,
    #     default="FedAvg",
    #     help="Algorithm list: FedAvg; FedOPT; FedProx ",
    # )

    # parser.add_argument(
    #     "--backend", type=str, default="MPI", help="Backend for Server and Client"
    # )

    # parser.add_argument(
    #     "--comm_round",
    #     type=int,
    #     default=10,
    #     help="how many round of communications we shoud use",
    # )

    # parser.add_argument(
    #     "--is_mobile",
    #     type=int,
    #     default=1,
    #     help="whether the program is running on the FedML-Mobile server side",
    # )

    # parser.add_argument(
    #     "--client_num_in_total",
    #     type=int,
    #     default=-1,
    #     metavar="NN",
    #     help="number of clients in a distributed cluster",
    # )

    # parser.add_argument(
    #     "--client_num_per_round",
    #     type=int,
    #     default=4,
    #     metavar="NN",
    #     help="number of workers",
    # )

    # parser.add_argument(
    #     "--epochs",
    #     type=int,
    #     default=3,
    #     metavar="EP",
    #     help="how many epochs will be trained locally",
    # )

    # parser.add_argument(
    #     "--gradient_accumulation_steps",
    #     type=int,
    #     default=1,
    #     metavar="EP",
    #     help="how many steps for accumulate the loss.",
    # )

    # parser.add_argument(
    #     "--client_optimizer",
    #     type=str,
    #     default="adam",
    #     help="Optimizer used on the client. This field can be the name of any subclass of the torch Opimizer class.",
    # )

    # parser.add_argument(
    #     "--learning_rate",
    #     type=float,
    #     default=0.1,
    #     metavar="LR",
    #     help="learning rate on the client (default: 0.001)",
    # )

    # parser.add_argument(
    #     "--weight_decay", type=float, default=0, metavar="N", help="L2 penalty"
    # )

    # parser.add_argument(
    #     "--clip_grad_norm", type=int, default=0, metavar="N", help="L2 penalty"
    # )

    # parser.add_argument(
    #     "--server_optimizer",
    #     type=str,
    #     default="sgd",
    #     help="Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.",
    # )

    # parser.add_argument(
    #     "--server_lr",
    #     type=float,
    #     default=0.1,
    #     help="server learning rate (default: 0.001)",
    # )

    # parser.add_argument(
    #     "--server_momentum", type=float, default=0, help="server momentum (default: 0)"
    # )

    # parser.add_argument(
    #     "--fedprox_mu", type=float, default=1, help="server momentum (default: 1)"
    # )

    # parser.add_argument(
    #     "--evaluate_during_training",
    #     default=False,
    #     metavar="EP",
    #     help="the frequency of the evaluation during training",
    # )

    # parser.add_argument(
    #     "--evaluate_during_training_steps",
    #     type=int,
    #     default=100,
    #     metavar="EP",
    #     help="the frequency of the evaluation during training",
    # )

    # parser.add_argument(
    #     "--frequency_of_the_test",
    #     type=int,
    #     default=1,
    #     help="the frequency of the algorithms",
    # )

    # # GPU device management
    # parser.add_argument(
    #     "--gpu_mapping_file",
    #     type=str,
    #     default="gpu_mapping.yaml",
    #     help="the gpu utilization file for servers and clients. If there is no \
    #                 gpu_util_file, gpu will not be used.",
    # )

    # parser.add_argument(
    #     "--gpu_mapping_key",
    #     type=str,
    #     default="mapping_default",
    #     help="the key in gpu utilization file",
    # )

    # parser.add_argument("--ci", type=int, default=0, help="CI")

    # # cached related
    # parser.add_argument(
    #     "--reprocess_input_data", action="store_true", help="whether generate features"
    # )

    # # freeze related
    # parser.add_argument(
    #     "--freeze_layers", type=str, default="", metavar="N", help="freeze which layers"
    # )

    # #  # UnifiedSKG used
    # # parser.add_argument('--seed', type=int, default=0, help='')
    # # parser.add_argument('--cfg', type=str, default='/u/tianshuzhang/FedNLP_spider/third_party_UnifiedSKG/configure/Salesforce/T5_base_finetune_spider_with_cell_value.cfg', help = '') 
    # # parser.add_argument('--run_name', type=str, default ='', help='')
    # # parser.add_argument('--logging_strategy', type=str, default ='steps', help = '') 
    # # parser.add_argument('--logging_first_step', default='true') 
    # # parser.add_argument('--logging_steps', type=int, default=0, help='')
    # # parser.add_argument('--evaluation_strategy', type=str, default ='steps', help = '') 
    # # parser.add_argument('--eval_steps', type=int, default=10, help='')
    # # parser.add_argument('--metric_for_best_model', default='eval_META_TUNING/spider_with_cell.cfg/exact_match') 
    # # parser.add_argument('--greater_is_better', default='true') 
    # # parser.add_argument('--save_strategy', type=str, default ='steps', help = '') 
    # # parser.add_argument('--save_steps', type=int, default=10, help='')
    # # parser.add_argument('--save_total_limit', type=int, default=10, help='')
    # # parser.add_argument('--load_best_model_at_end') 
    # # # parser.add_argument('--gradient_accumulation_steps', type=int, default=10, help='')
    # # parser.add_argument('--num_train_epochs', type=int, default=10, help='')
    # # parser.add_argument('--adafactor', default='true') 
    # # parser.add_argument('--learning_rate', type=float, default=5e-5, help='')
    # # parser.add_argument('--do_train') 
    # # parser.add_argument('--do_eval') 
    # # parser.add_argument('--do_predict') 
    # # parser.add_argument('--predict_with_generate') 
    # # # parser.add_argument('--output_dir') 
    # # parser.add_argument('--overwrite_output_dir') 
    # # parser.add_argument('--per_device_train_batch_size', type=int, default=0, help='')
    # # parser.add_argument('--per_device_eval_batch_size', type=int, default=0, help='')
    # # parser.add_argument('--generation_num_beams', type=int, default=1, help='')
    # # parser.add_argument('--generation_max_length', type=int, default=128, help='')
    # # parser.add_argument('--input_max_length', type=int, default=1024, help='')
    # # parser.add_argument('--ddp_find_unused_parameters', default='true') 
    # # parser.add_argument('--report_to', default='wandb') 
    # # parser.add_argument('--full_determinism', default='false') 
    # # parser.add_argument('--seed', default='2') 
    # # parser.add_argument('--skip_memory_metrics', default='true') 
    
    # # get_process_log_level

    # args = parser.parse_args("")
    # args.formulation = "spider"
    # dataset, class_num = load(args)



    # # print(dataset)
