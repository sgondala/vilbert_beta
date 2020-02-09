import argparse
import json
import logging
import os
import random
from io import open
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect
import yaml
from easydict import EasyDict as edict

import pdb
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_pretrained_bert.optimization import WarmupLinearSchedule

# from parallel.parallel import DataParallelModel, DataParallelCriterion

from vilbert.task_utils import LoadDatasets, LoadLosses, ForwardModelsTrain, ForwardModelsVal
from vilbert.optimization import BertAdam, Adam, Adamax
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

import vilbert.utils as utils
import torch.distributed as dist

from retreival_dataset import CiderDataset

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

writer = SummaryWriter()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers in the dataloader."
    )
    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="save name for training.", 
    )
    parser.add_argument(
        "--use_chunk", default=0, type=float, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--in_memory", default=False, type=bool, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--optimizer", default='BertAdam', type=str, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--tasks", default='', type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--freeze", default = -1, type=int, 
        help="till which layer of textual stream of vilbert need to fixed."
    )
    parser.add_argument(
        "--vision_scratch", action="store_true", help="whether pre-trained the image or not."
    )
    parser.add_argument(
        "--evaluation_interval", default=1, type=int, help="evaluate very n epoch."
    )
    parser.add_argument(
        "--lr_scheduler", default='mannul', type=str, help="whether use learning rate scheduler."
    )  
    parser.add_argument(
        "--baseline", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument(
        "--compact", action="store_true", help="whether use compact vilbert model."
    )

    args = parser.parse_args()
    assert len(args.output_dir) > 0

    with open('vlbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.load(f))

    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from vilbert.basebert import BaseBertForVLTasks
    elif args.compact:
        from vilbert.vilbert_compact import BertConfig
        from vilbert.vilbert_compact import VILBertForVLTasks        
    else:
        from vilbert.vilbert import BertConfig
        from vilbert.vilbert import VILBertForVLTasks

    if args.save_name:
        prefix = '-' + args.save_name
    else:
        prefix = ''
    timeStamp = '_' + args.config_file.split('/')[1].split('.')[0] + prefix
    savePath = os.path.join(args.output_dir, timeStamp)

    bert_weight_name = json.load(open("config/" + args.bert_model + "_weight_name.json", "r"))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    if default_gpu:
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    config = BertConfig.from_json_file(args.config_file)
    if default_gpu:
        # save all the hidden parameters. 
        with open(os.path.join(savePath, 'command.txt'), 'w') as f:
            print(args, file=f)  # Python 3.x
            print('\n', file=f)
            print(config, file=f)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    dataset = CiderDataset(
        'data/coco_generated_captions.json', 
        '/srv/share2/sgondala/tmp/trainval_36/python3_stuff/trainval_resnet101_faster_rcnn_genome_36.tsv', 
        'data/coco_cider_scores.json', 
        tokenizer,
        )

    length_of_data = len(dataset)

    train, val, test = random_split(dataset, [length_of_data - 20000, 5000, 15000])

    train_dataloader = DataLoader(train, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=256, shuffle=False)
 
    # task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, \
    #         task_dataloader_train, task_dataloader_val = LoadDatasets(args, task_cfg, args.tasks.split('-'))

    # tbLogger = utils.tbLogger(timeStamp, savePath, task_names, task_ids, task_num_iters, args.gradient_accumulation_steps)

    # if n_gpu > 0:
        # torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.baseline:
        model = BaseBertForVLTasks.from_pretrained(
            args.from_pretrained, config, num_labels=1, default_gpu=default_gpu
            )
    else:
        model = VILBertForVLTasks.from_pretrained(
            args.from_pretrained, config, num_labels=1, default_gpu=default_gpu
            )

    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)

    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if args.freeze != -1:
        bert_weight_name_filtered = []
        for name in bert_weight_name:
            if 'embeddings' in name:
                bert_weight_name_filtered.append(name)
            elif 'encoder' in name:
                layer_num = name.split('.')[2]
                if int(layer_num) <= args.freeze:
                    bert_weight_name_filtered.append(name)
        
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if key[12:] in bert_weight_name_filtered:
                value.requires_grad = False
            
        if default_gpu:
            print("filtered weight")
            print(bert_weight_name_filtered)

    optimizer_grouped_parameters = []
    lr = args.learning_rate
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'vil_prediction' in key:
                # if args.learning_rate <= 2e-5:
                lr = 1e-4
            else:
                if args.vision_scratch:
                    if key[12:] in bert_weight_name:
                        lr = args.learning_rate
                    else:
                        lr = 1e-4
                else:
                    lr = args.learning_rate
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.0}
                ]

    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    # max_num_iter = max(task_num_iters.values())
    # max_batch_size = max(task_batch_size.values())
    num_train_optimization_steps = (
        (200000//256) * args.num_train_epochs
    )

    if args.optimizer == 'BertAdam':
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            warmup=args.warmup_proportion,
            t_total=num_train_optimization_steps,
            schedule='warmup_constant',
        )
    elif args.optimizer == 'Adam':
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr=base_lr,
            warmup=args.warmup_proportion,
            t_total=num_train_optimization_steps,
            schedule='warmup_constant',
        )
    elif args.optimizer == 'Adamax':
        optimizer = Adamax(
            optimizer_grouped_parameters,
            lr=base_lr,
            warmup=args.warmup_proportion,
            t_total=num_train_optimization_steps,
            schedule='warmup_constant',
        )        

    if args.lr_scheduler == 'automatic':
        lr_scheduler = ReduceLROnPlateau(optimizer, \
                        mode='max',
                        factor=0.2, 
                        patience=1, 
                        cooldown=1,
                        threshold=0.001)
    elif args.lr_scheduler == 'mannul':
        lr_reduce_list = np.array([12, 16])
        # lr_reduce_list = np.array([6, 8, 10])        
        def lr_lambda_fun(epoch):
            return pow(0.1, np.sum(lr_reduce_list <= epoch))
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fun)

    if default_gpu:
        print("***** Running training *****")
        # print("  Num Iters: ", task_num_iters)
        # print("  Batch size: ", task_batch_size)
        # print("  Num steps: %d" %num_train_optimization_steps)

    criterion = nn.MSELoss()
    i = 0
    j = 0
    # initialize the data iteration.
    for epochId in tqdm(range(args.num_train_epochs), desc="Epoch"):
        model.train()
        for batch in train_dataloader:
            i += 1
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
            features, spatials, image_mask, captions, _, input_mask, segment_ids, co_attention_mask, image_id, y = batch
            _, vil_logit, _, _, _, _, _ = \
                model(captions, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)
            loss = torch.sqrt(criterion(vil_logit, y.to(device)))
            writer.add_scalar('Train loss', loss, i)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()

        model.eval()
        # when run evaluate, we run each task sequentially. 
        for batch in val_dataloader:
            j += 1
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
            features, spatials, image_mask, captions, _, input_mask, segment_ids, co_attention_mask, image_id, y = batch
            _, vil_logit, _, _, _, _, _ = \
                model(captions, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)
            loss = torch.sqrt(criterion(vil_logit, y.to(device)))
            writer.add_scalar('Val loss', loss, j)
        
        torch.save(model, args.save_path + "/model-" + str(epochId) + '.pth')

        # if args.lr_scheduler == 'automatic':
        #     lr_scheduler.step(ave_score)
        #     logger.info("best average score is %3f" %lr_scheduler.best)
        # else:
        lr_scheduler.step()

    # tbLogger.txt_close()
    
if __name__ == "__main__":
    main()