import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import Decoder


class nanoGPT(nn.Module):
    def __init__(self, vocab_size, ctxt_len, n_embed, num_heads, num_layers, dropout, device):
        super().__init__()

        self.vocab_size = vocab_size
        self.ctxt_len = ctxt_len
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device

        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(ctxt_len, n_embed)
        # self.decoders = [
        #     Decoder(n_embed, num_heads) for _ in range(num_layers)
        # ]
        self.decoder_array = nn.Sequential(
            *[Decoder(ctxt_len, n_embed, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def calc_loss(self, logits, targets):
        # logits shape: (batch_size*time_steps, vocab_size)
        logits = torch.reshape(logits, (-1, self.vocab_size))
        # targets shape: (batch_size*time_steps)
        targets = torch.reshape(targets, (-1,))
        loss = F.cross_entropy(logits, targets)
        return loss

    def forward(self, x):
        # x shape: (batch_size, time_steps)
        batch_size, time_steps = x.shape
        # token_embeds shape: (batch_size, time_steps, n_embed)
        token_embeds = self.token_embedding(x)
        # pos_embeds shape: (time_steps, n_embed)
        pos_embeds = self.pos_embedding(torch.arange(time_steps, device=self.device))
        # embeds shape: (batch_size, time_steps, n_embed)
        embeds = token_embeds + pos_embeds
        # decoder_out shape: (batch_size, time_steps, n_embed)
        decoder_out = self.decoder_array(embeds)
        # logits shape: (batch_size, time_steps, vocab_size)
        logits = self.lm_head(self.ln_f(decoder_out))

        return logits

    def generate(self, x, max_new_tokens):
        # x shape: (batch_size, time_steps)
        for _ in range(max_new_tokens):
            x_partial = x[:, -self.ctxt_len:]
            # logits shape: (batch_size, time_steps, vocab_size)
            logits = self(x_partial)
            # consider only the last time_step
            # logits shape: (batch_size, vocab_size)
            logits = logits[:, -1, :]
            # probs shape: (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # next_tokens shape: (batch_size, 1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # add next tokens to x
            x = torch.cat((x, next_tokens), dim=-1)

        return x
