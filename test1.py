########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################


"""guacamol:maxlen:100,scafold_max_len:100,vocab_size:95,n_embd:512"""
"""moses:maxlen:54,scafold_max_len:100,vocab_size:27,n_embd:256"""
"""BBBP:Max len:  232
Scaffold max len:  155,vocab_size:72,n_embd:256"""

'''/usr/local/miniconda3/envs/bighan2/bin/python /lab/Lxh/yan/Condition_Generation/molgpt-main/Pretrained-RWKV_TS/train.py --run_name guacamol_rwkv_cnn_sas_logp_tpsa --data_name guacamol2 --vocab_size 95 --batch_size 100 --max_len 100 --scaffold_maxlen 100'''
import logging
logging.basicConfig(level=logging.INFO)

import os

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer as Tra
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint


    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str, help="path of rwkv model")  # full path, with .pth
    parser.add_argument("--model_path", type=str, default=None, help="path of time series rwkv") # 
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=128, type=int)
    parser.add_argument("--epoch_steps", default=10000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=15, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

    parser.add_argument("--micro_bsz", default=100, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=8, type=int)

    parser.add_argument("--num_layers", default=3, type=int) #RNN layer
    parser.add_argument("--n_embd", default=512, type=int) # rnn 768 rwkv 512
    parser.add_argument("--dim_att", default=256, type=int)
    parser.add_argument("--dim_ffn", default=256, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_size_a", default=64, type=int)
    parser.add_argument("--head_size_divisor", default=4, type=int)

    parser.add_argument("--lr_init", default=8e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-6, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--dropout", default=0.05, type=float) # rwkv 0.05 rnn 0.3
    parser.add_argument("--weight_decay", default=0.001, type=float) # try 0.1 / 0.01 / 0.001
    parser.add_argument("--weight_decay_final", default=-1, type=float)

    parser.add_argument("--freeze_rwkv", default=0, type=int)  # layers to freeze
    parser.add_argument("--exp_name", default='dummy', type=str)  #


    parser.add_argument('--run_name', type=str,default="moses_no_prop_scaffold",
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    # in moses dataset, on average, there are only 5 molecules per scaffold
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--scaffold_maxlen',type=int,
                        default=100, help='condition on scaffold', required=False)
    parser.add_argument('--max_len',type=int,
                        default=100, help='condition on scaffold', required=False)
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--data_name', type=str, default='moses2',
                        help="name of the dataset to train on", required=False)
    parser.add_argument('--props', nargs="+", default=[],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
    parser.add_argument('--proteins', default=True,
                        help="properties to be used for condition", required=False)
    parser.add_argument('--pro_feat', type=int, default = 128, required=False)
   
    parser.add_argument('--batch_size', type=int, default=100,
                        help="batch size", required=False)
    
    if pl.__version__[0]=='2':
        parser.add_argument("--accelerator", default="gpu", type=str)
        parser.add_argument("--strategy", default="auto", type=str)
        parser.add_argument("--devices", default=2, type=int)
        parser.add_argument("--num_nodes", default=1, type=int)
        parser.add_argument("--precision", default="fp16", type=str)
        parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    else:
        parser = Tra.add_argparse_args(parser)

    args = parser.parse_args()

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    # 记录开始时间
    start_time = time.time()

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

    args.max_epochs = args.epoch_count  # continue forever
    args.betas = (args.beta1, args.beta2)
    # args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass


    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    import pandas as pd
    import numpy as np

    import torch

    import os, datetime
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer as Tra
    from dataset import UniSmileDataset
    from smiles_to_graph import GraphDataset,my_collate_fn
    import math
    import re
    from CharRNN import MyRNN
    from model_cnn_gnn_uni import MyRWKV
    from trainer import train_callback

    model = MyRWKV(args)
    # model = model.half()

    # checkpoint_path='/lab/Lxh/yan/checkpoints/best-model-epoch=04-val_loss=0.33.ckpt/checkpoint/mp_rank_00_model_states.pt'
    # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # # 从状态字典中提取模型参数
    # model_state_dict = checkpoint['module']
    # model.load_state_dict(model_state_dict)
    # model.load_state_dict(torch.load(f'/lab/Lxh/yan/Condition_Generation/molgpt-main/RWKV/{args.model_weight}.pt'))
    # model.to('cuda')
    # print('Model loaded')

    # 如果你想要更详细的信息，包括参数的名称
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.dtype}")
    if args.model_path:
        model.load_state_dict(torch.load(f'/lab/Lxh/yan/Condition_Generation/molgpt-main/RWKV1/{args.model_path}.pt'))
        model.to('cuda')
        print('Model loaded')
    if args.freeze_rwkv > 0:
        model.freeze_rwkv()


    data = pd.read_csv('/lab/Lxh/yan/Condition_Generation/molgpt-main/datasets/' + args.data_name + '.csv')
    data = data.dropna(axis=0).reset_index(drop=True)
    # data = data.sample(frac = 0.1).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    if 'moses' in args.data_name or 'zinc' in args.data_name or 'combined_uniprot' in args.data_name or 'antibiotics' in args.data_name:
        train_data = data[data['split'] == 'train'].reset_index(
            drop=True)   # 'split' instead of 'source' in moses
    else:
        train_data = data[data['source'] == 'train'].reset_index(
            drop=True)   # 'split' instead of 'source' in moses

    # train_data = train_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)

    if 'moses' in args.data_name or 'zinc' in args.data_name or 'combined_uniprot' in args.data_name or 'antibiotics' in args.data_name:
        val_data = data[data['split'] == 'test'].reset_index(
            drop=True)   # test for Moses. val for guacamol
    else:
        val_data = data[data['source'] == 'val'].reset_index(
            drop=True)   # test for Moses. val for guacamol

    # val_data = val_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)

    smiles = train_data['smiles']
    vsmiles = val_data['smiles']



    # prop = train_data[args.props].values.tolist()
    # vprop = val_data[args.props].values.tolist()
    # num_props = args.num_props

    # if args.scaffold:
    #     scaffold = train_data['scaffold_smiles']
    #     vscaffold = val_data['scaffold_smiles']
    
    if args.proteins:
        # 获取列名
        columns = [f'pro_feature{i}' for i in range(1, args.pro_feat+1)]
        proteins = train_data[columns].values.tolist()
        vproteins = val_data[columns].values.tolist()
        affinity=train_data['affinity'].values.tolist()
        vaffinity=val_data['affinity'].values.tolist()

    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip()))
              for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)
    print('Max len: ', max_len)
    args.max_len=max_len

    # lens = [len(regex.findall(i.strip()))
    #         for i in (list(scaffold.values) + list(vscaffold.values))]
    # scaffold_max_len = max(lens)
    # print('Scaffold max len: ', scaffold_max_len)
    # args.scaffold_maxlen=scaffold_max_len

    smiles = [i + str('<')*(args.max_len - len(regex.findall(i.strip())))
                for i in smiles]
    vsmiles = [i + str('<')*(args.max_len - len(regex.findall(i.strip())))
                for i in vsmiles]

    # scaffold = [i + str('<')*(args.scaffold_maxlen -
    #                             len(regex.findall(i.strip()))) for i in scaffold]
    # vscaffold = [i + str('<')*(args.scaffold_maxlen -
    #                             len(regex.findall(i.strip()))) for i in vscaffold]

    # whole_string = ' '.join(smiles + vsmiles + scaffold + vscaffold)
    whole_string = ' '.join(smiles + vsmiles)
    # whole_string = list(set(regex.findall(whole_string)))
    # print(whole_string,len(whole_string))
    # antibiotics 71 424 311
    # whole_string=['[nH]', '7', '[C@@]', '#', ')', '<', '[N+]', '[Fe+2]', '1', '\\', '[nH+]', '[PH]', 'S', '[C@@H]', '8', '6', '[Na+]', '[n-]', '[Na]', 'c', '[NH2+]', 'N', '[Tb+3]', '-', '[NH3+]', '[NH+]', 'P', '[Zn]', 'n', '[S@@]', '[Li+]', 's', '[SH+]', 'C', '.', 'Br', '[Fe+3]', '[C@]', '[Ag+]', '=', '5', 'o', '[P+]', '[O-]', '3', '[NH4+]', '[K+]', '%10', '[Mg+2]', '[C-]', '[Cl-]', '[n+]', 'O', '9', '[S+]', '(', '[Zn+2]', '4', '[Ag]', '[S@@+]', '[B-]', 'F', '2', '[C@H]', '[N-]', '[Ca+2]', 'B', '[S@]', 'Cl', '/', 'I']
    # BBBP
    # whole_string=['1', '%13', '4', '[o+]', '[Cl-]', '[PH]', '6', '%11', 'F', '[Na]', '.', '[n+]', '\\', 'O', '[Na+]', '[NH3+]', '-', 's', '[C@]', '[NH+]', '[C@H]', '7', '[nH]', 'Br', '#', 'Cl', 'I', '3', '[NH2+]', '[C@@H]', '[N+]', '8', '[Ca++]', '[N]', '[Cl]', 'P', '[S]', '[N@@]', '<', '=', '[S@]', '/', '%12', 'C', '[Br-]', '%14', '2', '[N-]', 'S', '[C@@]', '[nH+]', '[H]', 'N', '[P]', '[O+]', '[O-]', '(', '[H+]', '[NH-]', 'n', '[S+]', '5', '%10', '[OH-]', '[C-]', 'c', '[N@]', 'B', '9', '[NH]', ')', 'o']
    # zinc
    # whole_string=['o', '6', '[o+]', '/', '2', '[PH+]', '[SH]', 'C', ')', '#', '[nH]', '[PH]', 'S', 'I', '1', 'F', '[C@H]', '[P@]', '[nH+]', '[NH2+]', '[CH-]', '[s+]', 'Cl', '\\', '[PH2+]', '<', '[N+]', '4', '[n-]', 'P', 'O', '[S@@]', 'Br', 'n', '[NH3+]', '[S@]', '[NH-]', '[C@@]', '8', 'c', '7', '[SH+]', '[S+]', '[OH+]', '[S-]', '[P@@]', '[C@]', '-', '[N-]', '[O-]', 'N', '(', '[C@@H]', '[O+]', '[NH+]', 's', '[n+]', '[P@@H]', '[P+]', '5', '=', '[CH2-]', '[PH2]', '3', '[S@@+]']
    # guacamol
    # whole_string=[">","<",'[B-]', '[OH+]', '[nH+]', '[SiH2]', '[CH2+]', '[SeH+]', '[SH]', '[Si]', '-', '[s+]', '[BH2-]', 'Cl', '[cH-]', '[S-]', '[SeH]', '[CH+]', '[c+]', '[CH]', '[O-]', 'B', '[P+]', '1', '[IH2]', 's', '[N]', 'b', '[se+]', ')', '[n+]', '2', '9', 'P', '[O+]', '3', '[H]', '=', '[sH+]', '[PH]', '6', '#', 'c', 'I', '%10', '[SiH-]', '[CH2]', 'p', '[S+]', 'C', '7', '[se]', '[BH3-]', '[c-]', '[Si-]', 'o', '%11', '(', '%12', '[NH-]', 'n', 'S', '[Se+]', '[Se]', '[NH+]', '[cH+]', 'F', '[I+]', '[N-]', '[F+]', '[CH-]', '[bH-]', '[C+]', '[PH2+]', '5', '[NH3+]', '4', '[N+]', '[nH]', '[B]', 'Br', '[SiH]', '[O]', 'N', '[b-]', '[PH+]', '[BH-]', '[C-]', '[n-]', '8', '[o+]', 'O', '[IH]', '[NH2+]', '[SH+]']
    # whole_string=[">","<", "(", ")", "-", "1", "2", "3", "4", "5", "6", "#", "=", "Br", "C", "Cl", "F", "N", "O", "S", "[H]", "[SH]", "[nH]", "c", "n", "o", "s"]
    # proteins vocabsize=67 maxlen=84
    whole_string=['n', '3', 'Cl', '[CH+]', '(', 'P', '#', '[B]', '[CH2]', '[c+]', '[cH-]', '[C]', '[P+]', 'B', ')', 'o', 'N', '-', 's', '[CH-]', '[S-]', '[NH3+]', '[O+]', '[SH-]', '[N]', '[n+]', '[C-]', '[SH+]', '[nH]', '8', '=', '[n-]', '<', '[o+]', '[PH]', '[NH-]', '[O-]', '[NH+]', '4', 'F', '[c-]', '[NH2+]', '[CH2+]', '[N-]', '[O]', 'C', '[OH+]', '[B-]', '[s+]', '[BH3-]', '[nH+]', '[S+]', '[N+]', 'I', '7', 'O', '1', 'S', '[C+]', 'Br', '[S]', '9', '5', '[SH]', 'c', '2', '6']
    train_dataset = UniSmileDataset(args, smiles, whole_string, max_len, proteins=proteins,affinity=affinity)
    valid_dataset = UniSmileDataset(args, vsmiles, whole_string, max_len, proteins=vproteins,affinity=vaffinity)
    
    train_loader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=args.batch_size, num_workers=0, 
                             persistent_workers=False, drop_last=True)
    val_loader = DataLoader(valid_dataset, shuffle=False, pin_memory=True, batch_size=args.batch_size, num_workers=0, 
                           persistent_workers=False, drop_last=True)
    
    from pytorch_lightning.loggers import CSVLogger
    logger = CSVLogger("/lab/Lxh/yan/Condition_Generation/molgpt-main/Pretrained-RWKV_TS/logs", name=args.exp_name)

    current_datetime = datetime.datetime.now()
    # 设置 ModelCheckpoint 回调函数
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 监控的指标
        dirpath='checkpoints/',  # 保存模型的目录
        filename='best-model-{epoch:02d}-{val_loss:.2f}',  # 保存的文件名格式
        save_top_k=1,  # 保存最好的模型
        mode='min',  # 监控指标越小越好
        every_n_epochs=1,  # 每个 epoch 保存一次
    )

    trainer = Tra(max_epochs=args.epoch_count, logger=logger,  accelerator='gpu', devices=1, strategy="deepspeed_stage_1", precision='bf16',
                      callbacks=[train_callback(args),checkpoint_callback], enable_checkpointing=True)
    

    trainer.fit(model, train_loader, val_loader)

    # 保存模型的 state_dict
    torch.save(model.state_dict(), f'/lab/Lxh/yan/Condition_Generation/molgpt-main/RWKV2/{args.run_name}.pt')

    # 记录结束时间
    end_time = time.time()

    # 计算并打印运行时间
    elapsed_time = end_time - start_time
    print(f"程序运行时间：{elapsed_time} 秒")
