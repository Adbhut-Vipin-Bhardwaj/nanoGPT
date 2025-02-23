import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, ctxt_len, n_embed, dropout):
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.ctxt_len = ctxt_len
        self.n_embed = n_embed
        self.dropout = dropout

        self.register_buffer(
            'tril', torch.tril(torch.ones(ctxt_len, ctxt_len))
        )

        self.key = nn.Linear(n_embed, num_heads*head_size)
        self.query = nn.Linear(n_embed, num_heads*head_size)
        self.value = nn.Linear(n_embed, num_heads*head_size)
        self.wei_dropout = nn.Dropout(p=dropout)
        self.merge_heads = nn.Linear(num_heads*head_size, n_embed)
        self.out_dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x shape: (batch_size, time_steps, n_embed)
        batch_size, time_steps, n_embed = x.shape

        # keys shape: (batch_size, time_steps, num_heads*head_size)
        keys = self.key(x)
        # keys shape: (batch_size, time_steps, num_heads, head_size)
        keys = keys.reshape(
            batch_size, time_steps, self.num_heads, self.head_size
        )
        # keys shape: (batch_size, num_heads, time_steps, head_size)
        keys = torch.transpose(keys, 1, 2)

        # queries shape: (batch_size, time_steps, num_heads*head_size)
        queries = self.query(x)
        # keys shape: (batch_size, time_steps, num_heads, head_size)
        queries = queries.reshape(
            batch_size, time_steps, self.num_heads, self.head_size
        )
        # keys shape: (batch_size, num_heads, time_steps, head_size)
        queries = torch.transpose(queries, 1, 2)

        # keys shape: (batch_size, time_steps, num_heads*head_size)
        values = self.value(x)
        # keys shape: (batch_size, time_steps, num_heads, head_size)
        values = values.reshape(
            batch_size, time_steps, self.num_heads, self.head_size
        )
        # keys shape: (batch_size, num_heads, time_steps, head_size)
        values = torch.transpose(values, 1, 2)

        # wei shape: (batch_size, num_heads, time_steps, time_steps)
        wei = queries @ torch.transpose(keys, -2, -1) / (n_embed**0.5)
        wei = wei.masked_fill_(
            mask=self.tril[:time_steps, :time_steps] == 0,
            value=float('-inf')
        )
        wei = F.softmax(wei, dim=-1)
        wei = self.wei_dropout(wei)

        # concat_heads shape: (batch_size, num_heads, time_steps, head_size)
        concat_heads = wei @ values
        # concat_heads shape: (batch_size, time_steps, num_heads, head_size)
        concat_heads = concat_heads.transpose(1, 2)
        # concat_heads shape: (batch_size, time_steps, num_heads*head_size)
        concat_heads = concat_heads.reshape(batch_size, time_steps, -1)

        # out shape: (batch_size, time_steps, n_embed)
        out = self.merge_heads(concat_heads)
        out = self.out_dropout(out)

        return out
