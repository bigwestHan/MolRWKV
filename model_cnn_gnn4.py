########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, importlib
from torch_geometric.nn import GCNConv,GATConv,global_mean_pool,global_add_pool
from torch_geometric.loader import DataLoader
from torch.nn.utils.rnn import pad_sequence

import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
wkv6_cuda = load(name="wkv6", sources=["/lab/Lxh/yan/Condition_Generation/molgpt-main/Pretrained-RWKV_TS/cuda/wkv6_op.cpp", f"/lab/Lxh/yan/Condition_Generation/molgpt-main/Pretrained-RWKV_TS/cuda/wkv6_cuda.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
    
class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            r=r.to(torch.bfloat16)
            k=k.to(torch.bfloat16)
            v=v.to(torch.bfloat16)
            w=w.to(torch.bfloat16)
            u=u.to(torch.bfloat16)
            # assert r.dtype == torch.bfloat16
            # assert k.dtype == torch.bfloat16
            # assert v.dtype == torch.bfloat16
            # assert w.dtype == torch.bfloat16
            # assert u.dtype == torch.bfloat16
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################

class RWKV_Tmix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            if args.n_embd >= 4096:
                D_MIX_LORA = 64
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            D_DECAY_LORA = 64
            if args.n_embd >= 4096:
                D_DECAY_LORA = 128
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))
        # 确保 GroupNorm 层的权重和偏置也是 BFloat16 类型
        self.ln_x.weight = nn.Parameter(self.ln_x.weight.to(torch.bfloat16))
        self.ln_x.bias = nn.Parameter(self.ln_x.bias.to(torch.bfloat16))

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C).to(torch.bfloat16)
        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        r, k, v, g, w = self.jit_func(x)
        C=k.shape[-1]
        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.jit_func_2(x, g)

########################################################################################################

class RWKV_CMix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv
    
########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.ln3 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.conv = nn.Conv1d(args.n_embd, args.n_embd, kernel_size=3, padding=0)
        self.att = RWKV_Tmix_x060(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
        
    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)

        y=self.ln1(x)
        y=y.permute(0,2,1)
        y=F.pad(y, ( 2, 0, 0, 0), "constant", 0)
        y=torch.relu(self.conv(y))
        y=y.permute(0,2,1)
        x=x+y

        if self.args.dropout > 0:
            x=self.drop0(x)

        x = x + self.att(self.ln2(x))
        if self.args.dropout > 0:
            x=self.drop1(x)

        x = x + self.ffn(self.ln3(x))

        return x



class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

# 定义GCN模型
class MyGCN(torch.nn.Module):
    def __init__(self, num_features,n_embd):
        super(MyGCN, self).__init__()
        self.conv1 = GCNConv(num_features, n_embd)  # 第一层图卷积
        self.conv2 = GCNConv(n_embd, n_embd)  # 第二层图卷积
        # self.conv1 = GATConv(num_features, n_embd)  # 第一层GAT，多头注意力
        # self.conv2 = GATConv(n_embd, n_embd)  # 第二层GAT


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x=x.to(torch.bfloat16)
        # 三层图卷积
        x = F.relu(self.conv1(x, edge_index))
        x=F.dropout(x)
        x = self.conv2(x, edge_index)

        # 全局求和池化，以获得整个图的特征
        x = global_add_pool(x, data.batch)

        return x

def split_batch(full_batch_features, num_nodes_list):
    # 初始化一个空列表来存储分割后的图特征
    split_features_list = []

    # 初始化一个指针来追踪当前处理到的特征张量的哪个部分
    start_idx = 0

    # 遍历每个图的节点数
    for num_nodes in num_nodes_list:
        # 从full_batch_features中分割出当前图的特征
        end_idx = start_idx + num_nodes
        split_features = full_batch_features[start_idx:end_idx]

        # 将分割出的图特征添加到列表中
        split_features_list.append(split_features)

        # 更新指针位置
        start_idx = end_idx

    return split_features_list

class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        # if args.num_props:
        #     self.p_lin=nn.Linear(args.num_props,args.n_embd)
        if args.num_props==1:
            self.p_lin1=nn.Linear(1, args.n_embd)
        elif args.num_props==2:
            self.p_lin1=nn.Linear(1, args.n_embd)
            self.p_lin2=nn.Linear(1, args.n_embd)
        elif args.num_props==3:
            self.p_lin1=nn.Linear(1, args.n_embd)
            self.p_lin2=nn.Linear(1, args.n_embd)
            self.p_lin3=nn.Linear(1, args.n_embd)
        elif args.num_props==4:
            self.p_lin1=nn.Linear(1, args.n_embd)
            self.p_lin2=nn.Linear(1, args.n_embd)
            self.p_lin3=nn.Linear(1, args.n_embd)
            self.p_lin4=nn.Linear(1, args.n_embd)
        self.type_emb=nn.Embedding(2,args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.gnn = MyGCN(19,args.n_embd)


        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optim_groups = [{"params": trainable_params, "weight_decay": self.args.weight_decay}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, x, p=None, scaffold=None,data1=None):
        args = self.args
        # assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        b,t = x.size()


        type_embd = self.type_emb(torch.ones((b, 1), dtype = torch.long, device = x.device))
        x = self.emb(x) + type_embd

        if self.args.scaffold:
            data1=DataLoader(data1,batch_size=b)
            for d in data1:
                data1=d
                break
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = x.device))
            scaffold = self.emb(scaffold)+type_embd
            scaffold_graph = self.gnn(data1)

            scaffold_graph=scaffold_graph.unsqueeze(1)+type_embd
            x=torch.cat([scaffold_graph,scaffold,x],dim=1)

        if self.args.num_props:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = x.device))
            p=p.squeeze()
            p=p.unsqueeze(-1)
            
            if self.args.num_props==1:
                p = self.p_lin1(p).unsqueeze(1)
            elif self.args.num_props==2:
                p1=self.p_lin1(p[:,0,:])
                p2=self.p_lin2(p[:,1,:])
                p=torch.cat((p1.unsqueeze(1),p2.unsqueeze(1)),dim=1)
            elif self.args.num_props==3:
                p1=self.p_lin1(p[:,0,:])
                p2=self.p_lin2(p[:,1,:])
                p3=self.p_lin3(p[:,2,:])
                p=torch.cat((p1.unsqueeze(1),p2.unsqueeze(1),p3.unsqueeze(1)),dim=1)
            elif self.args.num_props==4:
                p1=self.p_lin1(p[:,0,:])
                p2=self.p_lin2(p[:,1,:])
                p3=self.p_lin3(p[:,2,:])
                p4=self.p_lin4(p[:,3,:])
                p=torch.cat((p1.unsqueeze(1),p2.unsqueeze(1),p3.unsqueeze(1),p4.unsqueeze(1)),dim=1)
            p = p+type_embd    # for single property
            # p = self.p_lin(p.unsqueeze(-1))+type_embd    # for single property

            x=torch.cat([p,x],dim=1)


        if args.dropout > 0:
            x = self.drop0(x)
        
        for block in self.blocks:
            if args.grad_cp == 1:
                x = deepspeed.checkpointing.checkpoint(block, x)
            else:
                x = block(x)

        # if self.args.scaffold:
        #     x=torch.cat([scaffold_graph,x],dim=1)

        x = self.ln_out(x)

        logits = self.head(x)

        if self.args.num_props and self.args.scaffold:
            num = int(self.args.num_props) +1+ int(self.args.scaffold_maxlen)
        elif self.args.num_props:
            num = int(self.args.num_props)
        elif self.args.scaffold:
            num = 1+int(self.args.scaffold_maxlen) 
        else:
            num = 0

        logits = logits[:, num:, :]

        return logits

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all


class MyRWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.rwkv = RWKV(args)
        if args.load_model:
            self.load_rwkv_from_pretrained(args.load_model)

        self.head = nn.Linear(args.n_embd, 1, bias=False)
        self.best_val_loss = torch.tensor(float("inf"))
        # self.prefix_len = args.prefix_len

    def load_rwkv_from_pretrained(self, path):
        self.rwkv.load_state_dict(torch.load(path, map_location="cpu"))
        rank_zero_info(f"Loaded pretrained RWKV from {path}")

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
    
    def freeze_rwkv(self):
        # freeze all layers excluding the layernorms, projection and head
        self.rwkv.emb.requires_grad_(False)
        self.rwkv.head.requires_grad_(False)
        for block in self.rwkv.blocks:
            for name, param in block.named_parameters():
                if "ln" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        name_of_trainable_params = [n for n, p in self.named_parameters() if p.requires_grad]
        rank_zero_info(f"Name of trainable parameters in optimizers: {name_of_trainable_params}")
        rank_zero_info(f"Number of trainable parameters in optimizers: {len(trainable_params)}")
        optim_groups = [{"params": trainable_params, "weight_decay": self.args.weight_decay}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    def forward(self, samples):
        
        x,p,scaffold,data1, y = samples["x"], samples["prop"], samples["scaffold"],samples["data1"], samples["y"]
        x = self.rwkv(x,p,scaffold,data1) # 
        return x, y

    def bidirectional_forward(self, x, x_emb=None):
        pass
    
    def training_step(self, batch, batch_idx):
        logits, targets = self(batch)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        print(f"Epoch {self.current_epoch} training loss: {loss}")
        return loss
    
    def on_training_epoch_end(self, outputs):
        all_losses = [x['loss'] for x in outputs if 'loss' in x]
        train_loss = sum(all_losses) / len(all_losses)
        self.log("train_loss", train_loss, sync_dist=True)
        my_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("my_lr", my_lr, sync_dist=True)
        print(f"Epoch {self.current_epoch} training loss: {train_loss}")
    
    def validation_step(self, batch, batch_idx):
        logits, targets = self(batch)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        self.log('val_loss', loss,  on_epoch=True, prog_bar=True)
        return loss
    
    # def on_validation_epoch_end(self):
    #     outputs = self.trainer.validated_outputs
    #     all_predicts = torch.cat([out["predicts"] for out in outputs])
    #     all_targets = torch.cat([out["targets"] for out in outputs])
    #     val_loss = F.cross_entropy(all_predicts.view(-1, all_predicts.size(-1)), all_targets.view(-1)).item()
    #     self.log("val_loss", val_loss, sync_dist=True)
    #     if val_loss < self.best_val_loss and self.trainer.current_epoch >=1:
    #         # remove last best model
    #         if os.path.exists(os.path.join(self.args.proj_dir, f"best-{self.best_val_loss:.3f}.pth")):
    #             os.remove(os.path.join(self.args.proj_dir, f"best-{self.best_val_loss:.3f}.pth"))
    #         state_dict = self.state_dict()
    #         torch.save(state_dict, os.path.join(self.args.proj_dir, f"best-{val_loss:.3f}.pth"))
    #         self.best_val_loss = val_loss
    #         # plot best figure
    #         from .utils import plot_prediction_and_target
    #         plot_dir = os.path.join(self.args.proj_dir, "best_model_plot")
    #         if not os.path.exists(plot_dir):
    #             os.makedirs(plot_dir)
    #         plot_prediction_and_target(outputs, plot_dir)
    
    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

    @torch.no_grad()
    def predict(self, input_points,p=None,sca=None,data1=None) -> list[int]:
        outputs = self.rwkv(input_points,p,sca,data1)
        return outputs
