import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttnScore(nn.Module):
    def __init__(self, hidden_size, attn_type='general'):
        assert attn_type in ['general', 'dot']
        self.attn_type = attn_type
        super(GlobalAttnScore, self).__init__()
        if attn_type == 'general':
            self.affine = nn.Linear(hidden_size, hidden_size)

    def forward(self, h_t, h_s):
        """
        h_t: batch x tgt_length x hidden_size
        h_s: batch x src_length x hidden_size
        returns: batch x tgt_length x src_length
        """
        if self.attn_type == 'general':
            h_t = self.affine(h_t)
        return torch.bmm(h_t, h_s.transpose(1, 2))


class Attention(nn.Module):

    def __init__(self, hidden_size, attn_type='general'):
        assert attn_type in ['general', 'dot']
        super(Attention, self).__init__()
        # not the final interface
        self.score = GlobalAttnScore(hidden_size, attn_type)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh()
        )

    def forward(self, input, context):
        """
        input: batch x tgt_length x hidden_size
        context: batch x src_length x hidden_size
        """
        tgt_batch, tgt_len, tgt_hidden = input.size()
        src_batch, src_len, src_hidden = context.size()
        attn_scores = self.score(input, context)
        alignment = F.softmax(attn_scores, dim=2)
        c = torch.bmm(alignment, context)

        attn_h_t = self.mlp(torch.cat([c, input], dim=2))
        return attn_h_t
