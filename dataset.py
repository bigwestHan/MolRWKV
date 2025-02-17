import torch
from torch.utils.data import Dataset
from utils import SmilesEnumerator
import numpy as np
import re
import math

class SmileDataset(Dataset):

    def __init__(self, args, data, content, block_size, aug_prob = 0.5, prop = None, scaffold = None, scaffold_maxlen = None):
        chars = content
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.args=args
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob
        

    def data_dropout(self, input_sequence, p=0.1, pk=0.3):
        """
        对输入序列应用Data Dropout。
        
        参数:
            input_sequence (torch.Tensor): 输入序列，形状为 (batch_size, sequence_length)。
            p (float): 每个token被丢弃的概率。
            pk (float): 应用Data Dropout的概率。
            
        返回:
            torch.Tensor: 应用了Data Dropout后的序列。
        """
        # 决定是否应用data dropout
        if torch.rand(1) < pk:
            # 为每个token生成一个随机矩阵，决定是否丢弃
            dropout_mask = torch.rand(input_sequence.shape) > p
            # 应用dropout_mask，丢弃的token将被设置为0
            dropped_input = torch.where(dropout_mask, input_sequence, torch.full_like(input_sequence, self.vocab_size-1))
            return dropped_input
        else:
            # 如果不应用data dropout，则直接返回原始序列
            return input_sequence
    
    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        smiles, prop, scaffold = self.data[idx], self.prop[idx], self.sca[idx]    # self.prop.iloc[idx, :].values  --> if multiple properties
        smiles = smiles.strip()
        scaffold = scaffold.strip()
       
        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
    

        smiles=regex.findall(smiles)

        # scaffold =str('>')+scaffold + str('<')*(self.scaf_max_len-1 - len(regex.findall(scaffold)))
        
        # if len(regex.findall(scaffold)) > self.scaf_max_len:
        #     scaffold = scaffold[:self.scaf_max_len]

        scaffold=regex.findall(scaffold)

        dix =  [self.stoi[s] for s in smiles]
        sca_dix = [self.stoi[s] for s in scaffold]

        sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        # x=self.data_dropout(x)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.long)
        prop = torch.tensor(prop, dtype = torch.float)

        # if self.args.num_props and self.args.scaffold:
        #     x=torch.cat([prop,sca_tensor,x],dim=-1)
        # elif self.args.num_props:
        #     x=torch.cat([prop,x],dim=-1)
        # elif self.args.scaffold:
        #     x=torch.cat([sca_tensor,x],dim=-1)


        return dict(x=x,prop=prop,sca_tensor=sca_tensor,y=y)


class UniSmileDataset(Dataset):

    def __init__(self, args, data, content, block_size, aug_prob = 0.5, proteins=None,affinity=None):
        chars = content
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
        self.proteins = proteins
        self.affinity = affinity
        self.args=args
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob
        

    def data_dropout(self, input_sequence, p=0.1, pk=0.3):
        """
        对输入序列应用Data Dropout。
        
        参数:
            input_sequence (torch.Tensor): 输入序列，形状为 (batch_size, sequence_length)。
            p (float): 每个token被丢弃的概率。
            pk (float): 应用Data Dropout的概率。
            
        返回:
            torch.Tensor: 应用了Data Dropout后的序列。
        """
        # 决定是否应用data dropout
        if torch.rand(1) < pk:
            # 为每个token生成一个随机矩阵，决定是否丢弃
            dropout_mask = torch.rand(input_sequence.shape) > p
            # 应用dropout_mask，丢弃的token将被设置为0
            dropped_input = torch.where(dropout_mask, input_sequence, torch.full_like(input_sequence, self.vocab_size-1))
            return dropped_input
        else:
            # 如果不应用data dropout，则直接返回原始序列
            return input_sequence
    
    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        smiles, proteins,affinity = self.data[idx], self.proteins[idx], self.affinity[idx] 
        smiles = smiles.strip()

        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
    

        smiles=regex.findall(smiles)


        dix =  [self.stoi[s] for s in smiles]

        x = torch.tensor(dix[:-1], dtype=torch.long)
        # x=self.data_dropout(x)
        y = torch.tensor(dix[1:], dtype=torch.long)
        proteins = torch.tensor(proteins, dtype = torch.float)
        affinity=torch.tensor(affinity, dtype = torch.float)


        return dict(x=x,proteins=proteins,affinity=affinity,y=y)
