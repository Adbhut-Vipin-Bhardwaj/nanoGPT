import torch.nn as nn

from multi_head_attention import MultiHeadAttention


class Decoder(nn.Module):
    def __init__(self, ctxt_len, n_embed, num_heads, dropout):
        super().__init__()

        self.n_embed = n_embed
        self.num_heads = num_heads
        self.dropout = dropout

        head_size = n_embed // num_heads
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.head_size = head_size

        self.ln1 = nn.LayerNorm(n_embed)
        self.self_attn = MultiHeadAttention(num_heads, head_size, ctxt_len, n_embed, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp1 = nn.Linear(n_embed, n_embed*4)
        self.mlp2 = nn.Linear(4*n_embed, n_embed)
        self.mlp_dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x shape: (batch_size, time_steps, n_embed)
        self_attn_out = x + self.self_attn(self.ln1(x))
        mlp_out = (
            self_attn_out
            + self.mlp_dropout(
                self.mlp2(self.mlp1(self.ln2(self_attn_out)))
            )
        )

        return mlp_out
