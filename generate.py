from utils import check_novelty, sample,proteins_sample,sample_gnn, canonic_smiles
from dataset import SmileDataset
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_mol,set_seed
import re
import json
from rdkit.Chem import RDConfig
import json

import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

import sascorer

from rdkit.Chem.rdMolDescriptors import CalcTPSA

from argparse import ArgumentParser
from pytorch_lightning import Trainer as Tra
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import pytorch_lightning as pl

#guacamol: 2
set_seed(12)

# /usr/local/miniconda3/envs/bighan2/bin/python /lab/Lxh/yan/Condition_Generation/molgpt-main/Pretrained-RWKV_TS/generate.py --model_weight rwkv_cnn1 --data_name moses2 --csv_name rwkv_cnn1 --vocab_size 27 --block_size 54
# python generate.py --model_weight moses_scaf_wholeseq_qed.pt --props qed --scaffold --data_name moses2 --csv_name debug --gen_size 1000 --vocab_size 94 --block_size 100

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
        parser.add_argument('--scaffold', action='store_true', default=False, help='condition on scaffold')
        parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
        parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=True)
        parser.add_argument('--data_name', type=str, default = 'moses2', help="name of the dataset to train on", required=False)
        parser.add_argument('--batch_size', type=int, default = 100, help="batch size", required=False)
        parser.add_argument('--gen_size', type=int, default = 10000, help="number of times to generate from a batch", required=False)
        parser.add_argument('--block_size', type=int, default = 54, help="number of layers", required=False)   # previously 57... 54 for moses. 100 for guacamol.
        parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
        parser.add_argument('--props', nargs="+", default = [], help="properties to be used for condition", required=False)
        parser.add_argument('--proteins', default=True,
                        help="properties to be used for condition", required=False)
        parser.add_argument('--pro_feat', type=int, default = 128, required=False)
        parser.add_argument('--rnn_layer', type=int, default=1, help="number of layers", required=False)
        parser.add_argument('--n_embd', type=int, default = 512, help="embedding dimension", required=False)
        parser.add_argument('--lstm_layers', type=int, default = 2, help="number of layers in lstm", required=False)
        parser.add_argument('--scaffold_maxlen',type=int,
                        default=232, help='condition on scaffold', required=False)
        parser.add_argument('--max_len',type=int,
                        default=232, help='condition on scaffold', required=False)

        parser.add_argument("--load_model", default="", type=str, help="path of rwkv model")  # full path, with .pth
        parser.add_argument("--model_path", type=str, default=None, help="path of time series rwkv") # 
        parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
        parser.add_argument("--random_seed", default="-1", type=int)

        parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

        parser.add_argument("--ctx_len", default=512, type=int)
        parser.add_argument("--epoch_steps", default=512, type=int)  # a mini "epoch" has [epoch_steps] steps
        parser.add_argument("--epoch_count", default=15, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
        parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
        parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

        parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
        parser.add_argument("--n_layer", default=8, type=int)

        parser.add_argument("--num_layers", default=3, type=int) #RNN layer
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
        parser.add_argument("--dropout", default=0.05, type=float) # try 0.01 / 0.02 / 0.05 / 0.1
        parser.add_argument("--weight_decay", default=0.001, type=float) # try 0.1 / 0.01 / 0.001
        parser.add_argument("--weight_decay_final", default=-1, type=float)
        parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough

        parser.add_argument("--freeze_rwkv", default=0, type=int)  # layers to freeze
        parser.add_argument("--exp_name", default='dummy', type=str)  #
        parser.add_argument("--label_smoothing", default=0, type=int)  # label smoothing window
        parser.add_argument("--prefix_len", default=0, type=int)  # 
        parser.add_argument("--shift_steps", default=5, type=int)  # shift steps for fj_windSpeed

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

        import warnings, math, datetime, sys, time
        from torch.utils.data import DataLoader
        if "deepspeed" in args.strategy:
                import deepspeed
        from pytorch_lightning import seed_everything

        if args.random_seed >= 0:
                print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
                seed_everything(args.random_seed)

        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
        warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
        # os.environ["WDS_SHOW_SEED"] = "1"

        args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        args.enable_checkpointing = False
        args.replace_sampler_ddp = False
        args.logger = False
        args.gradient_clip_val = 1.0
        args.num_sanity_val_steps = 0
        args.check_val_every_n_epoch = int(1e20)
        args.log_every_n_steps = int(1e20)
        args.max_epochs = args.epoch_count  # continue forever
        args.betas = (args.beta1, args.beta2)
        args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
        os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
        os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
        if args.dim_att <= 0:
                args.dim_att = args.n_embd
        if args.dim_ffn <= 0:
                args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size


        samples_per_epoch = args.epoch_steps * args.real_bsz
        tokens_per_epoch = samples_per_epoch * args.ctx_len
        try:
                deepspeed_version = deepspeed.__version__
        except:
                deepspeed_version = None
                pass
        rank_zero_info(
                f"""
        ############################################################################
        #
        # RWKV-5 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
        #
        # Data = {args.data_name} 
        #
        # Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
        #
        # Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
        #
        # Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
        #
        # Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
        #
        # Found torch {torch.__version__}, recommend 1.13.1+cu117 or newer
        # Found deepspeed {deepspeed_version}, recommend 0.7.0 (faster than newer versions)
        # Found pytorch_lightning {pl.__version__}, recommend 1.9.5
        #
        ############################################################################
        """
        )
        rank_zero_info(str(vars(args)) + "\n")


        assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
        os.environ["RWKV_FLOAT_MODE"] = args.precision
        if args.precision == "fp32":
                for i in range(10):
                        rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
        if args.precision == "fp16":
                rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

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
        from model_cnn_gnn_uni import MyRWKV

        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        

        context = "C"
        # context=context + str('<')*(args.max_len - len(regex.findall(context)))

        data = pd.read_csv('/lab/Lxh/yan/Condition_Generation/molgpt-main/datasets/'+args.data_name + '.csv')
        data = data.dropna(axis=0).reset_index(drop=True)
        data.columns = data.columns.str.lower()

        # if 'moses' in args.data_name:
        #     smiles = data[data['split']!='test_scaffolds']['smiles']   # needed for moses
        #     scaf = data[data['split']!='test_scaffolds']['scaffold_smiles']   # needed for moses
        # else:
        #     smiles = data[data['source']!='test']['smiles']
        #     scaf = data[data['source']!='test']['scaffold_smiles']

        # scaffold = data[data['split']!='test_scaffolds']['scaffold_smiles']
        # lens = [len(i.strip()) for i in scaffold.values]
        # scaffold_max_len = max(lens)

        # scaffold = data[data['split']=='test_scaffolds']['scaffold_smiles'].values
        # scaffold = sorted(list(scaffold))
        # condition = [scaffold[0], scaffold[len(scaffold)//2], scaffold[-1]]
        # condition = np.random.choice(scaffold, size = 3, replace = False)

        # condition = ['c1cnc2[nH]ccc2c1']
        # condition = ['O=C(CCc1cn[nH]c1)NCCC1CC2CCC1C2', 'O=C(CCC(=O)NCC1CCCO1)NCc1ccccc1', 'O=S(=O)(Cc1ccon1)NCc1cccs1']  # sim 0.9, 0.8, ~0.7


        #lens = [len(regex.findall(i)) for i in smiles]
        #max_len = max(lens)
        #smiles = [ i + str('<')*(max_len - len(regex.findall(i))) for i in smiles]

        #lens = [len(regex.findall(i)) for i in scaf]
        #scaffold_max_len = max(lens)
        
        #scaf = [ i + str('<')*(scaffold_max_len - len(regex.findall(i))) for i in scaf]
        if ('moses' in args.data_name) and args.scaffold:
            scaffold_max_len=54
        elif ('guacamol' in args.data_name) and args.scaffold:
            scaffold_max_len = 100
        elif ('BBBP' in args.data_name) and args.scaffold:
            scaffold_max_len = 232
        elif ('zinc' in args.data_name) and args.scaffold:
            scaffold_max_len = 72
        else:
            scaffold_max_len = 0
        args.scaffold_maxlen=scaffold_max_len
        #content = ' '.join(smiles + scaf)
        #chars = sorted(list(set(regex.findall(content))))

        #stoi = { ch:i for i,ch in enumerate(chars) }

        #with open(f'{args.data_name}_stoi.json', 'w') as f:
        #         json.dump(stoi, f)

        stoi = json.load(open(f'/lab/Lxh/yan/Condition_Generation/molgpt-main/{args.data_name}_stoi.json', 'r'))

        #itos = { i:ch for i,ch in enumerate(chars) }
        itos = { i:ch for ch,i in stoi.items() }

        #condition = ['O=C(Cc1ccccc1)NCc1ccccc1', 'c1cnc2[nH]ccc2c1', 'c1ccc(-c2ccnnc2)cc1', 'c1ccc(-n2cnc3ccccc32)cc1', 'O=C(c1cc[nH]c1)N1CCN(c2ccccc2)CC1']
        # condition = ['c1ccc(-n2cnc3ccccc32)cc1', 'O=C(c1cc[nH]c1)N1CCN(c2ccccc2)CC1']   # 'O=C(CNC(=O)NCCN1CCOCC1)Nc1ccccc1'
        #condition = [ i + str('<')*(scaffold_max_len - len(regex.findall(i))) for i in condition]
        #print(condition)

        num_props = len(args.props)

        model = MyRWKV(args)

        # checkpoint_path='/lab/Lxh/yan/checkpoints/best-model-epoch=05-val_loss=0.31-v2.ckpt/checkpoint/mp_rank_00_model_states.pt'
        # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # # 从状态字典中提取模型参数
        # model_state_dict = checkpoint['module']
        # model.load_state_dict(model_state_dict)
        model.load_state_dict(torch.load(f'/lab/Lxh/yan/Condition_Generation/molgpt-main/RWKV2/{args.model_weight}.pt'))
        model.to('cuda')
        print('Model loaded')

        # 计算总参数数量
        total_params = sum(p.numel() for p in model.parameters())

        print(f'Total number of parameters: {total_params}')

        gen_iter = math.ceil(args.gen_size / args.batch_size)
        # gen_iter = 2

        if 'guacamol' in args.data_name:
            prop2value = {
                        'qed': [0.5], 'sas': [3.0], 'logp': [4.0], 'tpsa': [40.0],
                        'tpsa_logp': [[40.0,2.0], [80.0,2.0], [120.0,2.0], [40.0,4.0], [80.0,4.0], [120.0,4.0], [40.0,6.0], [80.0,6.0], [120.0,6.0]],
                        # 'sas_logp': [[2.0, 2.0], [2.0, 4.0], [2.0, 6.0], [3.0, 2.0], [3.0, 4.0], [3.0, 6.0], [4.0, 2.0], [4.0, 4.0], [4.0, 6.0]],
                        # 'sas_tpsa': [[2.0,40.0], [2.0,80.0], [2.0,120.0], [3.0,40.0], [3.0,80.0], [3.0,120.0], [4.0,40.0], [4.0,80.0], [4.0,120.0]],
                        # 'tpsa_logp_sas': [[40.0, 2.0,2.0], [40.0, 2.0, 4.0], [40.0, 6.0, 2.0], [40.0, 6.0, 4.0], [80.0, 2.0, 2.0], [80.0, 2.0, 4.0], [80.0, 6.0, 2.0], [80.0, 6.0, 4.0]],
                        'logp_tpsa': [[4.0,80.0]],
                        'sas_logp': [[3.0,4.0]],
                        'sas_tpsa': [[3.0,80.0]],
                        'tpsa_logp_sas': [[80.0,4.0,3.0]],
                        }
        else:
            prop2value = {'qed': [0.6], 'sas': [3.0], 'logp': [4.0], 'tpsa': [30.0, 60.0, 90.0],
                        # 'tpsa_logp': [[40.0, 2.0], [80.0, 2.0], [40.0, 4.0], [80.0, 4.0]],
                        # 'sas_logp': [[2.0, 1.0], [2.0, 3.0], [3.5, 1.0], [3.5, 3.0]],
                        # 'tpsa_sas': [[40.0, 2.0], [80.0, 2.0], [40.0, 3.5], [80.0, 3.5]],
                        # 'sas_logp_tpsa': [[40.0, 1.0, 2.0], [40.0, 1.0, 3.5], [40.0, 3.0, 2.0], [40.0, 3.0, 3.5], [80.0, 1.0, 2.0], [80.0, 1.0, 3.5], [80.0, 3.0, 2.0], [80.0, 3.0, 3.5]]
                        'tpsa_logp_sas':[[80.0,4.0,3.0]],
                        'logp_tpsa':[[2.3794,40.54]]}
                        
            
        
        prop_condition = None
        if len(args.props) > 0:
            prop_condition = prop2value['_'.join(args.props)]
        
        scaf_condition = None

        if args.scaffold:
            scaf_condition = ['CC1COC2=C3N1C=C(C(=O)C3=CC(=C2N4CCN(CC4)C)F)C(=O)O ']
            # "O=C(N1CCc2c(F)ccc(F)c2C1)C1(O)Cc2ccccc2C1 ,O=C(C1Cc2ccccc2C1)N1CCc2ccccc2C1
        #     df = pd.read_csv('/lab/Lxh/yan/Condition_Generation/molgpt-main/datasets/zinc_gen_scaffold.csv')
        #     # 随机选择100个条目
        #     random_smiles = df.sample(n=100,random_state=6)
        #     # 将选中的scaffold_smiles保存到list中
        #     scaf_condition = random_smiles['scaffold_smiles'].tolist()
            scaf_condition = [ i + str('<')*(scaffold_max_len - len(regex.findall(i))) for i in scaf_condition]

        all_dfs = []
        all_metrics = []
        # for c in [0.3, 0.5, 0.7, 0.9]:
        # for c in [[2.0, 2.0], [2.0, 4.0], [2.0, 6.0], [3.0, 2.0], [3.0, 4.0], [3.0, 6.0], [4.0, 2.0], [4.0, 4.0], [4.0, 6.0]]:
        # for c in [[0.2, 2.0], [0.2, 3.0], [0.2, 4.0], [0.5, 2.0], [0.5, 3.0], [0.5, 4.0], [0.8, 2.0], [0.8, 3.0], [0.8, 4.0]]:
        # for c in [[0.3, 2.0, 2.0], [0.3, 2.0, 6.0], [0.3, 4.0, 6.0], [0.3, 4.0, 2.0], [0.8, 4.0, 6.0], [0.8, 2.0, 6.0], [0.8, 2.0, 2.0], [0.8, 4.0, 2.0]]:
        # for c in [[40.0, 2.0, 2.0], [40.0, 2.0, 4.0], [40.0, 6.0, 4.0], [40.0, 6.0, 2.0], [80.0, 6.0, 4.0], [80.0, 2.0, 4.0], [80.0, 2.0, 2.0], [80.0, 6.0, 2.0]]:
        # for c in [[40.0, 2.0], [80.0, 2.0], [120.0, 2.0], [40.0, 4.0], [80.0, 4.0], [120.0, 4.0], [40.0, 6.0], [80.0, 6.0], [120.0, 6.0]]:
        # for idx, temp in enumerate(np.arange(1.6, 1.7, 0.2)):
        # for j in condition:
                # for c in [[40.0, 2.0], [80.0, 2.0], [40.0, 4.0], [80.0, 4.0]]:  # tpsa + logp
                # for c in [1.0, 2.0, 3.0]:
                # for c in [0.6, 0.725, 0.85]:

        # for c in :
                # for c in [2.0, 2.75, 3.5]:
                # for c in [30.0, 60.0, 90.0]:
                # for c in [[40.0, 1.0, 2.0], [40.0, 1.0, 3.5], [40.0, 3.0, 2.0], [40.0, 3.0, 3.5], [80.0, 3.0, 3.5], [80.0, 1.0, 3.5], [80.0, 1.0, 2.0], [80.0, 3.0, 2.0]]:
                # for c in [[2.0, 1.0], [3.5, 1.0], [2.0, 3.0], [3.5, 3.0]]:   # sas + logp
                # for c in [[40.0, 2.0], [40.0, 3.5], [80.0, 2.0], [80.0, 3.5]]:   # tpsa + sas
                        # for c in [[2.0, 40.0, 2.0], [2.0, 40.0, 4.0], [6.0, 40.0, 4.0], [6.0, 40.0, 2.0], [6.0, 80.0, 4.0], [2.0, 80.0, 4.0], [2.0, 80.0, 2.0], [6.0, 80.0, 2.0]]:
                        # for c in [40.0, 80.0, 120.0]:

        data = pd.read_csv('/lab/Lxh/yan/Condition_Generation/molgpt-main/datasets/' + args.csv_name + '.csv')
        data = data.dropna(axis=0).reset_index(drop=True)
        # data = data.sample(frac = 0.1).reset_index(drop=True)
        data.columns = data.columns.str.lower()
        affinity=5.5
        columns = [f'pro_feature{i}' for i in range(1, args.pro_feat+1)]
        proteins=data[columns].values.tolist()
        count = 0
        if proteins is not None:
        #     for j in range(len(data)):
                molecules = []
                count += 1
                affinity=torch.tensor(affinity,dtype=torch.float32)
                proteins=torch.tensor(proteins[0],dtype=torch.float32)
                for i in tqdm(range(gen_iter)):
                        x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda') # 将张量在第一个维度上复制args.batch_size次，而在第二个维度上保持不变
                        affi=affinity.repeat(args.batch_size, 1).to('cuda')
                        pro=proteins.repeat(args.batch_size, 1).to('cuda')
                        y = proteins_sample(model, x,pro,affi, args.block_size, temperature=1, sample=True, top_k=None)   # 0.7 for guacamol
                        for gen_mol in y:
                                completion = ''.join([itos[int(i)] for i in gen_mol])
                                completion = completion.replace('<', '')
                                # gen_smiles.append(completion)
                                mol = get_mol(completion)
                                if mol:
                                        molecules.append(mol)

                "Valid molecules % = {}".format(len(molecules))

                mol_dict = []

                for i in molecules:
                        mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i),'target_seq':data['target_seq'].values.tolist()[0],'target_id':data['target_id'].values.tolist()[0],'true_affinity':affinity})
                print(len(mol_dict))
                results = pd.DataFrame(mol_dict)
                all_dfs.append(results)
        # if prop_condition is None and scaf_condition is None:
        #     molecules = []
        #     count += 1
        #     for i in tqdm(range(gen_iter)):
        #             x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda') # 将张量在第一个维度上复制args.batch_size次，而在第二个维度上保持不变
        #             p = None
        #             sca = None
        #             y = sample(model, x,p,sca, args.block_size, temperature=1, sample=True, top_k=None)   # 0.7 for guacamol
        #             for gen_mol in y:
        #                     completion = ''.join([itos[int(i)] for i in gen_mol])
        #                     completion = completion.replace('<', '')
        #                     # gen_smiles.append(completion)
        #                     mol = get_mol(completion)
        #                     if mol:
        #                             molecules.append(mol)

        #     "Valid molecules % = {}".format(len(molecules))

        #     mol_dict = []

        #     for i in molecules:
        #             mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})

        #     # for i in gen_smiles:
        #     #       mol_dict.append({'temperature' : temp, 'smiles': i})


        #     results = pd.DataFrame(mol_dict)

        #     # metrics = moses.get_all_metrics(gen_smiles)
        #     # metrics['temperature'] = temp

        #     # with open(f'gen_csv/moses_metrics_7_top10.json', 'w') as file:
        #     #       json.dump(metrics, file)

        #     canon_smiles = [canonic_smiles(s) for s in results['smiles']]
        #     unique_smiles = list(set(canon_smiles))
        #     if 'moses' in args.data_name or 'zinc' in args.data_name  or 'BBBP' in args.data_name:
        #             novel_ratio = check_novelty(unique_smiles, set(data['smiles']))   # replace 'source' with 'split' for moses
        #     else:
        #             novel_ratio = check_novelty(unique_smiles, set(data['smiles']))   # replace 'source' with 'split' for moses


        #     print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter), 3))
        #     print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
        #     print('Novelty ratio: ', np.round(novel_ratio/100, 3))

            
        #     results['qed'] = results['molecule'].apply(lambda x: QED.qed(x) )
        #     results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
        #     results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x) )
        #     results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x) )
        #     # results['temperature'] = temp
        #     results['validity'] = np.round(len(results)/(args.batch_size*gen_iter), 3)
        #     results['unique'] = np.round(len(unique_smiles)/len(results), 3)
        #     results['novelty'] = np.round(novel_ratio/100, 3)
        #     all_dfs.append(results)

        
        # elif (prop_condition is not None) and (scaf_condition is None):
        #     count = 0
        #     for c in prop_condition:
        #         molecules = []
        #         count += 1
        #         for i in tqdm(range(gen_iter)):
        #                 x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
        #                 p = None
        #                 if len(args.props) == 1:
        #                         p = torch.tensor([[c]]).repeat(args.batch_size, 1).to('cuda')   # for single condition
        #                 else:
        #                         p = torch.tensor(c).repeat(args.batch_size, 1).to('cuda')    # for multiple conditions
        #                 sca=None
        #                 y = sample(model, x,p,sca, args.block_size, temperature=1, sample=True, top_k=None)   # 0.7 for guacamol
        #                 for gen_mol in y:
        #                         completion = ''.join([itos[int(i)] for i in gen_mol])
        #                         completion = completion.replace('<', '')
        #                         # gen_smiles.append(completion)
        #                         mol = get_mol(completion)
        #                         if mol:
        #                                 molecules.append(mol)

        #         "Valid molecules % = {}".format(len(molecules))

        #         mol_dict = []

        #         for i in molecules:
        #                 mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})

        #         # for i in gen_smiles:
        #         #       mol_dict.append({'temperature' : temp, 'smiles': i})


        #         results = pd.DataFrame(mol_dict)

        #         # metrics = moses.get_all_metrics(gen_smiles)
        #         # metrics['temperature'] = temp

        #         # with open(f'gen_csv/moses_metrics_7_top10.json', 'w') as file:
        #         #       json.dump(metrics, file)

        #         canon_smiles = [canonic_smiles(s) for s in results['smiles']]
        #         unique_smiles = list(set(canon_smiles))
        #         if 'moses' in args.data_name or 'zinc' in args.data_name  or 'BBBP' in args.data_name:
        #                 novel_ratio = check_novelty(unique_smiles, set(data['smiles']))   # replace 'source' with 'split' for moses
        #         else:
        #                 novel_ratio = check_novelty(unique_smiles, set(data['smiles']))   # replace 'source' with 'split' for moses


        #         print(f'Condition: {c}')
        #         print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter), 3))
        #         print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
        #         print('Novelty ratio: ', np.round(novel_ratio/100, 3))

                
        #         if len(args.props) == 1:
        #                 results['condition'] = c
        #         elif len(args.props) == 2:
        #                 results['condition'] = str((c[0], c[1]))
        #         else:
        #                 results['condition'] = str((c[0], c[1], c[2]))
                        
        #         results['qed'] = results['molecule'].apply(lambda x: QED.qed(x) )
        #         results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
        #         results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x) )
        #         results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x) )
        #         # results['temperature'] = temp
        #         results['validity'] = np.round(len(results)/(args.batch_size*gen_iter), 3)
        #         results['unique'] = np.round(len(unique_smiles)/len(results), 3)
        #         results['novelty'] = np.round(novel_ratio/100, 3)
        #         results['species']=np.full(results['qed'].shape,count)
        #         all_dfs.append(results)


        # elif prop_condition is None and scaf_condition is not None:
        #     count = 0
        #     for j in scaf_condition:
        #         molecules = []
        #         count += 1
        #         for i in tqdm(range(gen_iter)):
        #             x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
        #             p=None
        #             sca = torch.tensor([stoi[s] for s in regex.findall(j)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
        #             sca_smiles=j.replace('<', '')
        #         #     y = sample(model, x,p,sca, args.block_size, temperature=1, sample=True, top_k=None)   # 0.7 for guacamol
        #             y = sample_gnn(model, x,p,sca,sca_smiles, args.block_size, temperature=1, sample=True, top_k=None)   # 0.7 for guacamol
        #             for gen_mol in y:
        #                     completion = ''.join([itos[int(i)] for i in gen_mol])
        #                     completion = completion.replace('<', '')
        #                     # gen_smiles.append(completion)
        #                     mol = get_mol(completion)
        #                     if mol:
        #                             molecules.append(mol)                                


        #         "Valid molecules % = {}".format(len(molecules))

        #         mol_dict = []

        #         for i in molecules:
        #                 mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})

        #         # for i in gen_smiles:
        #         #       mol_dict.append({'temperature' : temp, 'smiles': i})


        #         results = pd.DataFrame(mol_dict)

        #         # metrics = moses.get_all_metrics(gen_smiles)
        #         # metrics['temperature'] = temp

        #         # with open(f'gen_csv/moses_metrics_7_top10.json', 'w') as file:
        #         #       json.dump(metrics, file)

        #         canon_smiles = [canonic_smiles(s) for s in results['smiles']]
        #         unique_smiles = list(set(canon_smiles))
        #         if 'moses' in args.data_name or 'zinc' in args.data_name or 'BBBP' in args.data_name:
        #                 novel_ratio = check_novelty(unique_smiles, set(data['smiles']))   # replace 'source' with 'split' for moses
        #         else:
        #                 novel_ratio = check_novelty(unique_smiles, set(data['smiles']))   # replace 'source' with 'split' for moses

        #         j = j.replace('<', '')
        #         print(f'Scaffold: {j}')
        #         print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter), 3))
        #         print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
        #         print('Novelty ratio: ', np.round(novel_ratio/100, 3))

                
                        
        #         results['scaffold_cond'] = j
        #         results['qed'] = results['molecule'].apply(lambda x: QED.qed(x) )
        #         results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
        #         results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x) )
        #         results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x) )
        #         # results['temperature'] = temp
        #         results['validity'] = np.round(len(results)/(args.batch_size*gen_iter), 3)
        #         results['unique'] = np.round(len(unique_smiles)/len(results), 3)
        #         results['novelty'] = np.round(novel_ratio/100, 3)
        #         all_dfs.append(results)


        # elif prop_condition is not None and scaf_condition is not None:
        #     count = 0
        #     for j in scaf_condition:
        #         for c in prop_condition:
        #             molecules = []
        #             count += 1
        #             for i in tqdm(range(gen_iter)):
        #                 x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
        #                 p = None
        #                 if len(args.props) == 1:
        #                         p = torch.tensor([[c]]).repeat(args.batch_size, 1).to('cuda')   # for single condition
        #                 else:
        #                         p = torch.tensor([c]).repeat(args.batch_size, 1).unsqueeze(1).to('cuda')    # for multiple conditions
        #                 sca = torch.tensor([stoi[s] for s in regex.findall(j)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
        #                 sca_smiles=j.replace('<', '')
        #                 # y = sample(model, x,p,sca, args.block_size, temperature=1, sample=True, top_k=None)   # 0.7 for guacamol
        #                 y = sample_gnn(model, x,p,sca,sca_smiles, args.block_size, temperature=1, sample=True, top_k=None)   # 0.7 for guacamol
        #                 for gen_mol in y:
        #                         completion = ''.join([itos[int(i)] for i in gen_mol])
        #                         completion = completion.replace('<', '')
        #                         # gen_smiles.append(completion)
        #                         mol = get_mol(completion)
        #                         if mol:
        #                                 molecules.append(mol)                                


        #             "Valid molecules % = {}".format(len(molecules))

        #             mol_dict = []

        #             for i in molecules:
        #                     mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})

        #             # for i in gen_smiles:
        #             #       mol_dict.append({'temperature' : temp, 'smiles': i})


        #             results = pd.DataFrame(mol_dict)

        #             # metrics = moses.get_all_metrics(gen_smiles)
        #             # metrics['temperature'] = temp

        #             # with open(f'gen_csv/moses_metrics_7_top10.json', 'w') as file:
        #             #       json.dump(metrics, file)

        #             canon_smiles = [canonic_smiles(s) for s in results['smiles']]
        #             unique_smiles = list(set(canon_smiles))
        #             if 'moses' in args.data_name or 'zinc' in args.data_name or 'BBBP' in args.data_name:
        #                     novel_ratio = check_novelty(unique_smiles, set(data['smiles']))   # replace 'source' with 'split' for moses
        #             else:
        #                     novel_ratio = check_novelty(unique_smiles, set(data['smiles']))   # replace 'source' with 'split' for moses

        #             j = j.replace('<', '')
        #             print(f'Condition: {c}')
        #             print(f'Scaffold: {j}')
        #             print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter), 3))
        #             print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
        #             print('Novelty ratio: ', np.round(novel_ratio/100, 3))

                    
        #             if len(args.props) == 1:
        #                     results['condition'] = c
        #             elif len(args.props) == 2:
        #                     results['condition'] = str((c[0], c[1]))
        #             else:
        #                     results['condition'] = str((c[0], c[1], c[2]))
                            
        #             results['scaffold_cond'] = j
        #             results['qed'] = results['molecule'].apply(lambda x: QED.qed(x) )
        #             results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
        #             results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x) )
        #             results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x) )
        #             # results['temperature'] = temp
        #             results['validity'] = np.round(len(results)/(args.batch_size*gen_iter), 3)
        #             results['unique'] = np.round(len(unique_smiles)/len(results), 3)
        #             results['novelty'] = np.round(novel_ratio/100, 3)
        #             all_dfs.append(results)


        results = pd.concat(all_dfs)
        results.to_csv('/lab/Lxh/yan/Condition_Generation/molgpt-main/results1/' + args.csv_name + '.csv', index = False)

        # unique_smiles = list(set(results['smiles']))
        # canon_smiles = [canonic_smiles(s) for s in results['smiles']]
        # unique_smiles = list(set(canon_smiles))
        # if 'moses' in args.data_name or 'zinc' in args.data_name or 'BBBP' in args.data_name:
        #         novel_ratio = check_novelty(unique_smiles, set(data['smiles']))    # replace 'source' with 'split' for moses
        # else:
        #         novel_ratio = check_novelty(unique_smiles, set(data['smiles']))    # replace 'source' with 'split' for moses
               

        # print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter*count), 3))
        # print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
        # print('Novelty ratio: ', np.round(novel_ratio/100, 3))

