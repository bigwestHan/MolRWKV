import json, time, random, os
import numpy as np
import dataclasses
from torch.nn import functional as F
from typing import List, Dict
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt

time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt


def plot_prediction_and_target(outputs, output_dir):
    for i in range(len(outputs)):
        plt.figure()
        pred = outputs[i]["predicts"].cpu().float().numpy()
        target = outputs[i]["targets"].cpu().float().numpy()
        x = np.arange(len(pred))
        plt.plot(x, pred, label="prediction", zorder=2)
        plt.plot(x, target, label="target", zorder=1)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"day_{i}.png"))
        plt.close()
    # plot week prediction
    week_outputs = [outputs[i:i+7] for i in range(0, len(outputs), 7)]
    week_pred = [np.concatenate([x["predicts"].cpu().float().numpy() for x in week], axis=0) for week in week_outputs]
    week_target = [np.concatenate([x["targets"].cpu().float().numpy() for x in week], axis=0) for week in week_outputs]
    for i in range(len(week_pred)):
        plt.figure()
        x = np.arange(len(week_pred[i]))
        plt.plot(x, week_pred[i], label="prediction", zorder=2)
        plt.plot(x, week_target[i], label="target", zorder=1)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"week_{i}.png"))
        plt.close()
    # plot month prediction
    month_outputs = [outputs[i:i+30] for i in range(0, len(outputs), 30)]
    month_pred = [np.concatenate([x["predicts"].cpu().float().numpy() for x in month], axis=0) for month in month_outputs]
    month_target = [np.concatenate([x["targets"].cpu().float().numpy() for x in month], axis=0) for month in month_outputs]
    for i in range(len(month_pred)):
        plt.figure()
        x = np.arange(len(month_pred[i]))
        plt.plot(x, month_pred[i], label="prediction", zorder=2)
        plt.plot(x, month_target[i], label="target", zorder=1)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"month_{i}.png"))
        plt.close()

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
# from moses.utils import get_mol
from rdkit import Chem
   
import numpy as np
import threading

from rdkit import Chem
def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

# @torch.no_grad()
# def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, prop = None, scaffold = None):
#     """
#     take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
#     the sequence, feeding the predictions back into the model each time. Clearly the sampling
#     has quadratic complexity unlike an RNN that is only linear, and has a finite context window
#     of block_size, unlike an RNN that has an infinite context window.
#     """
#     block_size = model.get_block_size()   
#     model.eval()

#     for k in range(steps):
#         x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
#         logits, _, _ = model(x_cond, prop = prop, scaffold = scaffold)   # for liggpt
#         # logits, _, _ = model(x_cond)   # for char_rnn
#         # pluck the logits at the final step and scale by temperature
#         logits = logits[:, -1, :] / temperature
#         # optionally crop probabilities to only the top k options
#         if top_k is not None:
#             logits = top_k_logits(logits, top_k)
#         # apply softmax to convert to probabilities
#         probs = F.softmax(logits, dim=-1)
#         # sample from the distribution or take the most likely
#         if sample:
#             ix = torch.multinomial(probs, num_samples=1)
#         else:
#             _, ix = torch.topk(probs, k=1, dim=-1)
#         # append to the sequence and continue
#         x = torch.cat((x, ix), dim=1)

#     return x

def check_novelty(gen_smiles, train_smiles): # gen: say 788, train: 120803
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]  # [1]*45
        novel = len(gen_smiles) - sum(duplicates)  # 788-45=743
        novel_ratio = novel*100./len(gen_smiles)  # 743*100/788=94.289
    print("novelty: {:.3f}%".format(novel_ratio))
    return novel_ratio

def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

    #Experimental Class for Smiles Enumeration, Iterator and SmilesIterator adapted from Keras 1.2.2

class Iterator(object):
    """Abstract base class for data iterators.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)
        if n < batch_size:
            raise ValueError('Input data length is shorter than batch_size\nAdjust batch_size')

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)




class SmilesIterator(Iterator):
    """Iterator yielding data from a SMILES array.
    # Arguments
        x: Numpy array of SMILES input data.
        y: Numpy array of targets data.
        smiles_data_generator: Instance of `SmilesEnumerator`
            to use for random SMILES generation.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        dtype: dtype to use for returned batch. Set to keras.backend.floatx if using Keras
    """

    def __init__(self, x, y, smiles_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dtype=np.float32
                 ):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        self.x = np.asarray(x)

        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.smiles_data_generator = smiles_data_generator
        self.dtype = dtype
        super(SmilesIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + [ self.smiles_data_generator.pad, self.smiles_data_generator._charlen]), dtype=self.dtype)
        for i, j in enumerate(index_array):
            smiles = self.x[j:j+1]
            x = self.smiles_data_generator.transform(smiles)
            batch_x[i] = x

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


class SmilesEnumerator(object):
    """SMILES Enumerator, vectorizer and devectorizer
    
    #Arguments
        charset: string containing the characters for the vectorization
          can also be generated via the .fit() method
        pad: Length of the vectorization
        leftpad: Add spaces to the left of the SMILES
        isomericSmiles: Generate SMILES containing information about stereogenic centers
        enum: Enumerate the SMILES during transform
        canonical: use canonical SMILES during transform (overrides enum)
    """
    def __init__(self, charset = '@C)(=cOn1S2/H[N]\\', pad=120, leftpad=True, isomericSmiles=True, enum=True, canonical=False):
        self._charset = None
        self.charset = charset
        self.pad = pad
        self.leftpad = leftpad
        self.isomericSmiles = isomericSmiles
        self.enumerate = enum
        self.canonical = canonical

    @property
    def charset(self):
        return self._charset
        
    @charset.setter
    def charset(self, charset):
        self._charset = charset
        self._charlen = len(charset)
        self._char_to_int = dict((c,i) for i,c in enumerate(charset))
        self._int_to_char = dict((i,c) for i,c in enumerate(charset))
        
    def fit(self, smiles, extra_chars=[], extra_pad = 5):
        """Performs extraction of the charset and length of a SMILES datasets and sets self.pad and self.charset
        
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
            extra_chars: List of extra chars to add to the charset (e.g. "\\\\" when "/" is present)
            extra_pad: Extra padding to add before or after the SMILES vectorization
        """
        charset = set("".join(list(smiles)))
        self.charset = "".join(charset.union(set(extra_chars)))
        self.pad = max([len(smile) for smile in smiles]) + extra_pad
        
    def randomize_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        return Chem.MolToSmiles(nm, canonical=self.canonical, isomericSmiles=self.isomericSmiles)

    def transform(self, smiles):
        """Perform an enumeration (randomization) and vectorization of a Numpy array of smiles strings
        #Arguments
            smiles: Numpy array or Pandas series containing smiles as strings
        """
        one_hot =  np.zeros((smiles.shape[0], self.pad, self._charlen),dtype=np.int8)
        
        if self.leftpad:
            for i,ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                l = len(ss)
                diff = self.pad - l
                for j,c in enumerate(ss):
                    one_hot[i,j+diff,self._char_to_int[c]] = 1
            return one_hot
        else:
            for i,ss in enumerate(smiles):
                if self.enumerate: ss = self.randomize_smiles(ss)
                for j,c in enumerate(ss):
                    one_hot[i,j,self._char_to_int[c]] = 1
            return one_hot

      
    def reverse_transform(self, vect):
        """ Performs a conversion of a vectorized SMILES to a smiles strings
        charset must be the same as used for vectorization.
        #Arguments
            vect: Numpy array of vectorized SMILES.
        """       
        smiles = []
        for v in vect:
            #mask v 
            v=v[v.sum(axis=1)==1]
            #Find one hot encoded index with argmax, translate to char and join to string
            smile = "".join(self._int_to_char[i] for i in v.argmax(axis=1))
            smiles.append(smile)
        return np.array(smiles)





import math

# lxh
class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        _x=sublayer(self.norm(x))
        return x + self.dropout(_x[0])
    

@torch.no_grad()
def sample(model, x,p,sca, steps, temperature=1.0, sample=False, top_k=None):
    """
    model: 预训练模型，如 GPT 或类似的生成模型。
    x: 输入的条件序列的索引，形状为 (batch_size, sequence_length)。
    steps: 生成的步数，即需要生成的新标记数量。
    temperature: 温度参数，用于调整生成文本的多样性。
    sample: 布尔值，指示是否从概率分布中进行采样。如果为 True，则从分布中随机采样一个标记；如果为 False，则选择概率最高的标记。
    top_k: 可选参数，如果指定，则仅保留概率最高的前 top_k 个选项。
    prop 和 scaffold: 可选的其他参数，传递给模型调用的附加参数。
    """
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = steps   # 获取模型的块大小，用于限制上下文的长度
    model.eval() # 将模型设为评估模式，确保不进行训练调整

    for k in range(steps): # 循环 steps 次，生成指定数量的标记
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed 如果输入序列 x 的长度超过 block_size，则截断为最后的 block_size 长度
        logits = model.predict(x_cond,p,sca)   # for liggpt 使用给定的条件 x_cond，模型生成下一个标记的概率分布
        # logits, _, _ = model(x_cond)   # for char_rnn
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature # 将模型生成的 logits （对数概率）按温度参数进行缩放
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1) # 将 logits 转换为概率分布，使用 softmax 函数
        # sample from the distribution or take the most likely
        if sample: # 如果 sample=True，则使用 torch.multinomial() 从概率分布 probs 中进行多项式采样,随机采样，返回一个采样后的标记
            ix = torch.multinomial(probs, num_samples=1)
        else: # 如果 sample=False，则使用 torch.topk() 选择概率最高的标记
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1) # 将新生成的标记 ix 追加到原始输入序列 x 的末尾，以便下一步预测使用

    return x

@torch.no_grad()
def proteins_sample(model, x,proteins,affinity, steps, temperature=1.0, sample=False, top_k=None):
    """
    model: 预训练模型，如 GPT 或类似的生成模型。
    x: 输入的条件序列的索引，形状为 (batch_size, sequence_length)。
    steps: 生成的步数，即需要生成的新标记数量。
    temperature: 温度参数，用于调整生成文本的多样性。
    sample: 布尔值，指示是否从概率分布中进行采样。如果为 True，则从分布中随机采样一个标记；如果为 False，则选择概率最高的标记。
    top_k: 可选参数，如果指定，则仅保留概率最高的前 top_k 个选项。
    prop 和 scaffold: 可选的其他参数，传递给模型调用的附加参数。
    """
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = steps   # 获取模型的块大小，用于限制上下文的长度
    model.eval() # 将模型设为评估模式，确保不进行训练调整

    for k in range(steps): # 循环 steps 次，生成指定数量的标记
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed 如果输入序列 x 的长度超过 block_size，则截断为最后的 block_size 长度
        logits = model.predict(x_cond,proteins,affinity)   # for liggpt 使用给定的条件 x_cond，模型生成下一个标记的概率分布
        # logits, _, _ = model(x_cond)   # for char_rnn
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature # 将模型生成的 logits （对数概率）按温度参数进行缩放
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1) # 将 logits 转换为概率分布，使用 softmax 函数
        # sample from the distribution or take the most likely
        if sample: # 如果 sample=True，则使用 torch.multinomial() 从概率分布 probs 中进行多项式采样,随机采样，返回一个采样后的标记
            ix = torch.multinomial(probs, num_samples=1)
        else: # 如果 sample=False，则使用 torch.topk() 选择概率最高的标记
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1) # 将新生成的标记 ix 追加到原始输入序列 x 的末尾，以便下一步预测使用

    return x

from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.utils import one_hot, scatter

types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4,'Br':5,'Cl':6,'S':7,'P':8,'I':9,'B':10,'Si':11,'Se':12}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

def mol_to_data(mol):
        N = mol.GetNumAtoms()

        type_idx = []
        atomic_number = []
        aromatic = []
        sp = []
        sp2 = []
        sp3 = []
        num_hs = []
        for atom in mol.GetAtoms():
            type_idx.append(types[atom.GetSymbol()])
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
            edge_types += 2 * [bonds[bond.GetBondType()]]

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_attr = one_hot(edge_type, num_classes=len(bonds))

        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        edge_attr = edge_attr[perm]

        row, col = edge_index
        hs = (z == 1).to(torch.float)
        num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

        x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
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

@torch.no_grad()
def sample_gnn(model, x,p,sca,scaffold_smiles, steps, temperature=1.0, sample=False, top_k=None):
    """
    model: 预训练模型，如 GPT 或类似的生成模型。
    x: 输入的条件序列的索引，形状为 (batch_size, sequence_length)。
    steps: 生成的步数，即需要生成的新标记数量。
    temperature: 温度参数，用于调整生成文本的多样性。
    sample: 布尔值，指示是否从概率分布中进行采样。如果为 True，则从分布中随机采样一个标记；如果为 False，则选择概率最高的标记。
    top_k: 可选参数，如果指定，则仅保留概率最高的前 top_k 个选项。
    prop 和 scaffold: 可选的其他参数，传递给模型调用的附加参数。
    """
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = steps   # 获取模型的块大小，用于限制上下文的长度
    model.eval() # 将模型设为评估模式，确保不进行训练调整
    if scaffold_smiles is None:
        datas=None
    else:
        mol = Chem.MolFromSmiles(scaffold_smiles)
        data = mol_to_data(mol).to(x.device)
        datas=[data for _ in range(x.size(0))]

    for k in range(steps): # 循环 steps 次，生成指定数量的标记
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed 如果输入序列 x 的长度超过 block_size，则截断为最后的 block_size 长度
        logits= model.predict(x_cond, p,sca,datas)   # for liggpt 使用给定的条件 x_cond，模型生成下一个标记的概率分布
        # logits, _, _ = model(x_cond)   # for char_rnn
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature # 将模型生成的 logits （对数概率）按温度参数进行缩放
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1) # 将 logits 转换为概率分布，使用 softmax 函数
        # sample from the distribution or take the most likely
        if sample: # 如果 sample=True，则使用 torch.multinomial() 从概率分布 probs 中进行多项式采样,随机采样，返回一个采样后的标记
            ix = torch.multinomial(probs, num_samples=1)
        else: # 如果 sample=False，则使用 torch.topk() 选择概率最高的标记
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # 替换第i列
        # x[:, k+1] = ix.squeeze()  # 使用squeeze()去掉ix的单维度，以匹配x的列
        x = torch.cat((x, ix), dim=1) # 将新生成的标记 ix 追加到原始输入序列 x 的末尾，以便下一步预测使用

    return x