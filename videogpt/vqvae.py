import argparse
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .affine import AffineTransform
from .attention import MultiHeadAttention
from .utils import shift_dim


class VQVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.n_codes = args.n_codes
        self.encoder = Encoder(args.n_hiddens, args.n_res_layers, args.downsample)
        self.decoder = Decoder(args.n_hiddens, args.n_res_layers, args.downsample)
        self.pre_vq_conv = SamePadConv3d(args.n_hiddens, args.embedding_dim, 1)
        self.post_vq_conv = SamePadConv3d(args.embedding_dim, args.n_hiddens, 1)
        self.codebook = Codebook(
            args.n_codes,
            args.embedding_dim,
            beta=getattr(args, "beta", 0.25),
            affine_lr=getattr(args, "affine_lr", 0.0),
            affine_groups=getattr(args, "affine_groups", 1),
            use_running_statistics=getattr(args, "use_running_statistics", False),
            shared_codes_ratio=getattr(args, "shared_codes_ratio", 0.0),
            top_k_experts=getattr(args, "top_k_experts", 1),
        )
        self.save_hyperparameters()
        # Initialize a list to store validation step outputs
        self.validation_step_outputs = []

    @property
    def latent_shape(self):
        input_shape = (
            self.args.sequence_length,
            self.args.resolution,
            self.args.resolution,
        )
        return tuple([s // d for s, d in zip(input_shape, self.args.downsample)])

    def encode(self, x, include_embeddings=False):
        h = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(h)
        if include_embeddings:
            return vq_output["encodings"], vq_output["embeddings"]
        else:
            return vq_output["encodings"]

    def decode(self, encodings):
        h = F.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output["embeddings"]))
        # reconstruction_loss = | | x - x_recon | |² / (2σ²), 重建损失。
        recon_loss = F.mse_loss(x_recon, x) / 0.06
        return recon_loss, x_recon, vq_output

    def training_step(self, batch, batch_idx):
        x = batch["video"]
        recon_loss, _, vq_output = self.forward(x)
        commitment_loss = vq_output["commitment_loss"]
        loss = recon_loss + commitment_loss
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["video"]
        recon_loss, _, vq_output = self.forward(x)

        # Instead of logging directly, store the results
        result = {
            "val_recon_loss": recon_loss,
            "val_perplexity": vq_output["perplexity"],
            "val_commitment_loss": vq_output["commitment_loss"],
            "batch_size": x.size(0),  # Store batch size for weighted average
        }

        # Store the results for epoch-end processing
        self.validation_step_outputs.append(result)

        return result

    def on_validation_epoch_end(self):
        # Calculate statistics across all validation batches
        if not self.validation_step_outputs:
            return

        # Calculate total samples across all batches
        total_samples = sum(out["batch_size"] for out in self.validation_step_outputs)

        # Calculate weighted averages
        avg_recon_loss = (
            sum(
                out["val_recon_loss"] * out["batch_size"]
                for out in self.validation_step_outputs
            )
            / total_samples
        )
        avg_perplexity = (
            sum(
                out["val_perplexity"] * out["batch_size"]
                for out in self.validation_step_outputs
            )
            / total_samples
        )
        avg_commitment_loss = (
            sum(
                out["val_commitment_loss"] * out["batch_size"]
                for out in self.validation_step_outputs
            )
            / total_samples
        )

        # Log the epoch-level metrics
        self.log("val/recon_loss", avg_recon_loss, prog_bar=True)
        self.log("val/perplexity", avg_perplexity, prog_bar=True)
        self.log("val/commitment_loss", avg_commitment_loss, prog_bar=True)

        # Clear the outputs list to free memory
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embedding_dim", type=int, default=256)
        parser.add_argument("--n_codes", type=int, default=2048)
        parser.add_argument("--n_hiddens", type=int, default=240)
        parser.add_argument("--n_res_layers", type=int, default=4)
        parser.add_argument("--downsample", nargs="+", type=int, default=(4, 4, 4))
        parser.add_argument("--beta", type=float, default=0.25)
        parser.add_argument("--affine_lr", type=float, default=0.0)
        parser.add_argument("--affine_groups", type=int, default=1)
        parser.add_argument("--use_running_statistics", action="store_true")
        parser.add_argument("--shared_codes_ratio", type=float, default=0.0, 
                           help="Ratio of codebook entries to be used as shared codes (0.0-1.0)")
        parser.add_argument("--top_k_experts", type=int, default=1)
        return parser


class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(
            shape=(0,) * 3,
            dim_q=n_hiddens,
            dim_kv=n_hiddens,
            n_head=n_head,
            n_layer=1,
            causal=False,
            attn_type="axial",
        )
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2), **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3), **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4), **kwargs)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x


class AttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            SamePadConv3d(n_hiddens, n_hiddens // 2, 3, bias=False),
            nn.BatchNorm3d(n_hiddens // 2),
            nn.ReLU(),
            SamePadConv3d(n_hiddens // 2, n_hiddens, 1, bias=False),
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
            AxialBlock(n_hiddens, 2),
        )

    def forward(self, x):
        return x + self.block(x)


class Codebook(nn.Module):
    def __init__(
        self,
        n_codes,
        embedding_dim,
        beta=0.25,
        affine_lr=0.0,
        affine_groups=1,
        use_running_statistics=False,
        shared_codes_ratio=0.0,
        top_k_experts=1,
    ):
        super().__init__()
        self.register_buffer("_embeddings", torch.randn(n_codes, embedding_dim))
        self.register_buffer("N", torch.zeros(n_codes))
        self.register_buffer("z_avg", self._embeddings.data.clone())
        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.beta = beta
        
        # Track shared and expert-specific codes
        self.shared_codes_ratio = shared_codes_ratio
        self.n_shared_codes = int(n_codes * shared_codes_ratio)
        self.n_expert_codes = n_codes - self.n_shared_codes
        self.top_k_experts = top_k_experts
        
        # Register a buffer to track shared codes usage
        if self.n_shared_codes > 0:
            self.register_buffer("is_shared_code", torch.zeros(n_codes, dtype=torch.bool))
            # Mark the first n_shared_codes as shared
            self.is_shared_code[:self.n_shared_codes] = True
            
        
        # Add affine transformation support
        if affine_lr > 0:
            self.affine_transform = AffineTransform(
                embedding_dim,
                use_running_statistics=use_running_statistics,
                lr_scale=affine_lr,
                num_groups=affine_groups,
            )

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)
        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self._embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    @property
    def embeddings(self):
        """Property that returns the potentially transformed codebook entries"""
        codebook = self._embeddings
        if hasattr(self, "affine_transform"):
            codebook = self.affine_transform(codebook)
        return codebook

    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)

        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)

        # Get potentially transformed codebook
        codebook = self.embeddings

        # Update affine statistics if needed
        if hasattr(self, "affine_transform"):
            self.affine_transform.update_running_statistics(
                flat_inputs, self._embeddings
            )

        # 实现共享codebook逻辑
        if self.n_shared_codes > 0:
            # 1. 将codebook分为shared codes和expert-specific codes
            shared_codebook = codebook[:self.n_shared_codes]
            expert_codebook = codebook[self.n_shared_codes:]
            
            # 2. 计算与全部共享codes的embedding
            # 为每个shared code计算一个权重
            # 计算输入与shared codes的相似度，使用点积
            similarities = flat_inputs @ shared_codebook.t()  # [batch, n_shared_codes]
            
            # 将相似度转换为权重
            shared_weights = F.softmax(similarities, dim=1)  # [batch, n_shared_codes]
            
            # 使用权重计算共享codes的加权贡献
            shared_emb_contributions = torch.matmul(shared_weights, shared_codebook)  # [batch, embedding_dim]
            
            # 计算每个输入与所有shared codes的距离 (仅用于EMA更新统计)
            shared_distances = (
                (flat_inputs**2).sum(dim=1, keepdim=True)
                - 2 * flat_inputs @ shared_codebook.t()
                + (shared_codebook.t() ** 2).sum(dim=0, keepdim=True)
            )
            
            # 3. 创建one-hot表示用于统计信息更新
            # 为了EMA更新，我们需要记录哪些shared codes被使用
            shared_encoding_indices = torch.argmin(shared_distances, dim=1)
            shared_onehot = F.one_hot(shared_encoding_indices, self.n_shared_codes).type_as(flat_inputs)
            
            # 4. 计算与expert-specific codes的相似度score
            expert_scores = flat_inputs @ expert_codebook.t()  # [batch, n_expert_codes]
            
            # 5. 选择Top-K的expert (使用配置的top_k_experts参数)
            K = min(self.top_k_experts, self.n_expert_codes)  # 确保K不超过可用的expert数量
            topk_scores, topk_indices = torch.topk(expert_scores, k=K, dim=1)  # [batch, K]
            
            # 6. 归一化得分为权重 (使用sigmoid函数，与图中公式匹配)
            expert_weights = torch.sigmoid(topk_scores)  # [batch, K]
            expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)  # [batch, K]
            
            # 7. 记录expert的one-hot表示用于EMA更新
            expert_onehot = torch.zeros(
                flat_inputs.shape[0], self.n_expert_codes, device=flat_inputs.device, dtype=flat_inputs.dtype
            )
            batch_indices = torch.arange(flat_inputs.shape[0], device=flat_inputs.device).repeat_interleave(K)
            expert_indices_flat = topk_indices.reshape(-1)
            expert_weights_flat = expert_weights.reshape(-1)
            
            # 8. 在expert_onehot中将选中的expert位置设为权重值
            expert_onehot.index_put_(
                (batch_indices, expert_indices_flat),
                expert_weights_flat,
                accumulate=True
            )
            
            # 9. 通过embedding lookup获取expert embeddings
            selected_expert_embeddings = F.embedding(
                topk_indices, expert_codebook
            )  # [batch, K, embedding_dim]
            
            # 10. 将expert embeddings与权重相乘并求和
            weighted_expert_emb = (selected_expert_embeddings * expert_weights.unsqueeze(-1)).sum(dim=1)  # [batch, embedding_dim]
            
            # 11. 合并shared codes和expert codes的效果
            # 根据图中公式：h't = ut + sum(FFN(s)(ut)) + sum(gi,t * FFN(r)(ut))
            # 在VQVAE的上下文中，我们将其解释为:
            # result = input + sum of shared contributions + weighted sum of expert contributions
            final_embeddings = flat_inputs + shared_emb_contributions + weighted_expert_emb
            
            # 12. 创建完整的one-hot编码用于统计信息更新
            # 将shared codes的onehot和expert codes的onehot拼接在一起
            encode_onehot = torch.cat([
                shared_onehot,  # [batch, n_shared_codes]
                expert_onehot   # [batch, n_expert_codes]
            ], dim=1)
            
            # 13. 为了与VQVAE的其余部分兼容，我们需要提供一个indices作为编码
            # 选择贡献最大的expert作为编码索引
            max_expert_indices = topk_indices[:, 0]  # 使用第一个（或唯一的）expert索引
            encoding_indices = max_expert_indices + self.n_shared_codes
            encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])
            
            # 14. 为了与VQVAE的其余部分兼容，重新整形final_embeddings
            embeddings = final_embeddings.reshape(z.shape[0], -1, *z.shape[2:])  # [b, embedding_dim, t, h, w]
            embeddings = shift_dim(embeddings, 1, 1)  # 调整维度顺序
        else:
            # 原始距离计算方法，不使用shared codes
            distances = (
                (flat_inputs**2).sum(dim=1, keepdim=True)
                - 2 * flat_inputs @ codebook.t()
                + (codebook.t() ** 2).sum(dim=0, keepdim=True)
            )
            encoding_indices = torch.argmin(distances, dim=1)
            encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)
            encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])
            
            # 使用编码索引查找embeddings
            embeddings = F.embedding(encoding_indices, codebook)
            embeddings = shift_dim(embeddings, -1, 1)

        # Commitment loss with beta parameter
        commitment_loss = self.beta * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self._embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            # Handle dead codes differently for shared and expert-specific codes
            if self.n_shared_codes > 0:
                # For shared codes (higher priority to keep active)
                shared_usage = (self.N[:self.n_shared_codes].view(self.n_shared_codes, 1) >= 1).float()
                self._embeddings[:self.n_shared_codes].data.mul_(shared_usage).add_(
                    _k_rand[:self.n_shared_codes] * (1 - shared_usage)
                )
                
                # For expert-specific codes 
                expert_usage = (self.N[self.n_shared_codes:].view(self.n_expert_codes, 1) >= 1).float()
                self._embeddings[self.n_shared_codes:].data.mul_(expert_usage).add_(
                    _k_rand[self.n_shared_codes:] * (1 - expert_usage)
                )
            else:
                # Original behavior without shared codes
                usage = (self.N.view(self.n_codes, 1) >= 1).float()
                self._embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        # Straight-through estimator
        embeddings_st = (embeddings - z).detach() + z

        # Calculate perplexity for all codes
        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return dict(
            embeddings=embeddings_st,
            encodings=encoding_indices,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings

    def get_affine_params(self):
        if hasattr(self, "affine_transform"):
            return self.affine_transform.get_affine_params()
        return None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Encoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, downsample):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = nn.ModuleList()
        max_ds = n_times_downsample.max()
        in_channels = None
        for i in range(max_ds):
            in_channels = 3 if i == 0 else n_hiddens
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, n_hiddens, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(in_channels, n_hiddens, kernel_size=3)

        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens) for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU()
        )

    def forward(self, x):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
        h = self.conv_last(h)
        h = self.res_stack(h)
        return h

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Decoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, upsample):
        super().__init__()
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens) for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU()
        )

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = SamePadConvTranspose3d(n_hiddens, out_channels, 4, stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1

    def forward(self, x):
        h = self.res_stack(x)
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        return h

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias
        )

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
            padding=tuple([k - 1 for k in kernel_size]),
        )

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
