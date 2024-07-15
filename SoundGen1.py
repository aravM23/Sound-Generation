import math
import random

import joblib
import numpy as np
import torch
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, EinMix
from encodec import EncodecModel
from torch import nn
import torch.nn.functional as F
from core.conformer import Conformer

class SoundStorm(nn.Module):
    def __init__(self, dim=1024, heads=16, linear_units=4096, num_blocks=12, 
                 semantic_codebook_size=1024, semantic_num_quantizers=1, 
                 acoustic_codebook_size=1024, acoustic_num_quantizers=8, 
                 positionwise_conv_kernel_size=5, encodec=None, hubert_kmean_path=None):
        super(SoundStorm, self).__init__()
        self.dim = dim
        self.heads = heads
        self.linear_units = linear_units
        self.num_blocks = num_blocks
        self.semantic_codebook_size = semantic_codebook_size
        self.semantic_num_quantizers = semantic_num_quantizers
        self.acoustic_codebook_size = acoustic_codebook_size
        self.acoustic_num_quantizers = acoustic_num_quantizers
        self.positionwise_conv_kernel_size = positionwise_conv_kernel_size
        self.encodec = encodec
        self.hubert_kmean_path = hubert_kmean_path

        self.mask_token_id = num_codes_with_mask
        self.mask_upper_level = num_codes_with_mask

        self.sos_tokens = sos_token

        self.lm = Conformer(
            attention_dim=dim,
            attention_heads=heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size
        )

        self.heads = nn.Sequential(
            nn.Linear(dim, dim * acoustic_num_quantizers),
            Rearrange('b n (h d) -> b (n h) d', h=acoustic_num_quantizers),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12),
            Rearrange('b (n q) d -> b n q d', q=acoustic_num_quantizers)
        )

        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(num_codes_with_mask + 2)) for _ in range(acoustic_num_quantizers)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            EinMix(
                'b n q d -> b n q l',
                weight_shape='q d l',
                bias_shape='q l',
                q=acoustic_num_quantizers,
                l=num_codes_with_mask + 2,
                d=dim
            )
        )

        if encodec is not None:
            self._read_embedding_from_encodec(encodec)

        if hubert_kmean_path is not None:
            self._read_embedding_from_hubert_kmeans(hubert_kmean_path)

    def masking(self, codes, q=None, t=None):
        seq_len = codes.shape[1]
        batch = codes.shape[0]
        codes = rearrange(codes, 'b n q -> q b n')

        masked_codes = []

        for i, code in enumerate(codes):
            if q == i:
                c, label = self.level_mask(code, seq_len, batch, t, codes.device)
                masked_codes.append(c)
            elif i > q:
                masked_codes.append(self.fine_mask(code, t))
            else:
                masked_codes.append(code)

        return masked_codes, label

    def forward(self, cond, codes, return_loss=True):
        """
        cond: [B, Len]
        codes: [B, N_q, Len]
        """

        b, q, n = codes.shape

        q = random.randint(0, self.acoustic_num_quantizers - 1)
        t = random.randint(0, codes.shape[1] - 1)

        masked_codes, labels = self.masking(codes, q, t)

        masked_codes = torch.stack(masked_codes, dim=0)
        masked_codes = rearrange(masked_codes, 'q b n -> b n q')

        emb = None

        for i, layer in enumerate(self.code_embeds):
            if emb is None:
                emb = layer(masked_codes[:, :, i].unsqueeze(-1)).squeeze(-2)
            else:
                emb = emb + layer(masked_codes[:, :, i].unsqueeze(-1)).squeeze(-2)

        semb = self.semantic_embeds(cond)               
        semb = self.sem_cond_proj(semb)

        emb = emb + semb

        out, _ = self.lm(emb, None)                        

        out = self.heads(out)                        

        logits = self.to_logits(out)            

    def tokens_to_logits(self, semb, input_codes):
        emb = semb
        for i, layer in enumerate(self.code_embeds):
            emb = emb + layer(input_codes[:, :, i])

        out, _ = self.lm(emb, None)   # [B, n, d]
        out = self.heads(out)         # [B, q*n, d]
        logits = self.to_logits(out)  # [B, n, q, d]

        return logits

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to("cuda")
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        masking = (confidence < cut_off)
        return masking

    @torch.no_grad()
    def generate(self, conds, codes):
        num_latents_input = int(conds.size(1))
        num_prompt = min(int(num_latents_input * 0.5), 225)

        iter_steps = self.steps[rvq_layer]
        times = torch.linspace(0., 1., iter_steps + 1)
        all_mask_num_tokens = (cosine_schedule(times[1:]) * seq_len).long()

        for mask_num_tokens, steps_until_x0 in zip(all_mask_num_tokens.tolist(), reversed(range(iter_steps))):
            # ... (see below)

            logits = self.tokens_to_logits(semb, torch.cat([prompt, seq], dim=1))
            logits = logits.view(batch_size, num_latents_to_generate + num_prompt, 8, 1025)
            logits = logits[:, num_prompt:, rvq_layer, :]

            logits = top_k(logits, self.filter_threshold)
            sampled_ids = gumbel_sample(logits, temperature=max(self.temperature, 1e-3))

            seq[:, :, rvq_layer] = torch.where(mask[:, :, rvq_layer], sampled_ids, seq[:, :, rvq_layer])

            # ... (see below)

            scores = 1 - logits.softmax(dim=-1)
            scores = scores.gather(2, rearrange(sampled_ids, 'b n -> b n 1'))
            scores = rearrange(scores, 'b n 1 -> b n')

            if mask_num_tokens == 0:
                continue

            scores = scores.masked_fill(~mask[:, :, rvq_layer], -torch.finfo(scores.dtype).max)
            mask_indices = scores.topk(mask_num_tokens, dim=-1).indices
            mask[:, :, rvq_layer] = torch.zeros_like(scores, dtype=torch.bool).scatter(1, mask_indices, True)
            seq[:, :, rvq_layer] = seq[:, :, rvq_layer].masked_fill(mask[:, :, rvq_layer], self.mask_token_id)

        out = torch.cat([prompt, seq], dim=1)
        return out