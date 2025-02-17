import torch
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem import Draw
from torch_geometric.utils import one_hot, scatter
import numpy as np
import math
import re
from utils import SmilesEnumerator


class MyData():
    def __init__(self, prop, data1, data2, y):
        self.prop=prop
        self.data1=data1
        self.data2=data2
        self.y=y

    
# 定义一个自定义的 collate_fn 函数
def my_collate_fn(batch):
    # 假设每个数据项都是 MyData 实例
    prop = [item['prop'] for item in batch if item is not None]
    scaffold=[item['scaffold'] for item in batch]
    data1=[item['data1'] for item in batch]
    x=[item['x'] for item in batch]
    y=[item['y'] for item in batch]
    
    # 将张量列表堆叠成一个批次张量
    prop = torch.stack(prop)
    scaffold = torch.stack(scaffold)
    x = torch.stack(x)
    y=torch.stack(y)
    
    
    return dict(prop=prop,scaffold=scaffold, data1=data1, x=x, y=y)

class GraphDataset(torch.utils.data.Dataset):

    def __init__(self, args, data, content, block_size, aug_prob = 0.5, prop = None, scaffold = None, scaffold_maxlen = None):
        data_size, vocab_size = len(data), len(content)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = { ch:i for i,ch in enumerate(content) }
        self.itos = { i:ch for i,ch in enumerate(content) }
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob
        self.types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4,'Br':5,'Cl':6,'S':7,'P':8,'I':9,'B':10,'Si':11,'Se':12}
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    
    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)
    
    def mol_to_data(self, mol):
        N = mol.GetNumAtoms()

        type_idx = []
        atomic_number = []
        aromatic = []
        sp = []
        sp2 = []
        sp3 = []
        num_hs = []
        for atom in mol.GetAtoms():
            type_idx.append(self.types[atom.GetSymbol()])
            atomic_number.append(atom.GetAtomicNum())
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            sp.append(1 if hybridization == HybridizationType.SP else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

        z = torch.tensor(atomic_number, dtype=torch.long)

        rows, cols, edge_types = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows += [start, end]
            cols += [end, start]
            edge_types += 2 * [self.bonds[bond.GetBondType()]]

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_attr = one_hot(edge_type, num_classes=len(self.bonds))

        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        edge_attr = edge_attr[perm]

        row, col = edge_index
        hs = (z == 1).to(torch.float)
        num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

        x1 = one_hot(torch.tensor(type_idx), num_classes=len(self.types))
        x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                            dtype=torch.float).t().contiguous()
        x = torch.cat([x1, x2], dim=-1)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

        data = Data(
            x=x,
            edge_index=edge_index,
            smiles=smiles,
            edge_attr=edge_attr,
            num_nodes_list=N
        )

        return data
    
    def __getitem__(self, idx):
        prop, scaffold, smiles = self.prop[idx], self.sca[idx], self.data[idx]
        smiles = smiles.strip()
        scaffold = scaffold.strip()

        sca=scaffold.replace('<', '')
        # 将 SMILES 转换mol
        mol1 = Chem.MolFromSmiles(sca)
        if mol1 is None:
            return None
        data1 = self.mol_to_data(mol1)
        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        smiles=regex.findall(smiles)
        scaffold=regex.findall(scaffold)

        dix =  [self.stoi[s] for s in smiles]
        sca_dix = [self.stoi[s] for s in scaffold]

        scaffold = torch.tensor(sca_dix, dtype=torch.long)

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        prop = torch.tensor([prop], dtype = torch.float)

        data_group = (prop, data1, x, y)

        return dict(prop=prop,scaffold=scaffold, data1=data1, x=x, y=y)