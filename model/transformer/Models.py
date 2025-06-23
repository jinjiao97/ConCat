import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.transformer.Constants as Constants
from model.transformer.Layers import EncoderLayer



def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask



class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, device, t_scale, global_attention):
        super().__init__()
        self.d_model = d_model
        self.t_scale = t_scale
        self.global_attention = global_attention
        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / (d_model//2)) for i in range(d_model//2)],
            device=device)

        self.layer_stack1 = nn.ModuleList([
            EncoderLayer(d_model//2, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        self.layer_stack2 = nn.ModuleList([
            EncoderLayer(d_model//2, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec * self.t_scale
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, enc_output, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        if self.global_attention:
            slf_attn_mask = get_attn_key_pad_mask(seq_k=non_pad_mask, seq_q=non_pad_mask)
        else:
            slf_attn_mask_subseq = get_subsequent_mask(non_pad_mask)
            slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=non_pad_mask, seq_q=non_pad_mask)
            slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
            slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        non_pad_mask = torch.unsqueeze(non_pad_mask, dim=-1)
        tem_enc = self.temporal_enc(event_time, non_pad_mask)

        enc_output1 = enc_output[:, :, :self.d_model//2] + tem_enc
        enc_output2 = enc_output[:, :, self.d_model//2:]

        for enc_layer in self.layer_stack1:
            enc_output1, _ = enc_layer(
                enc_output1,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        for enc_layer in self.layer_stack2:
            enc_output2, _ = enc_layer(
                enc_output2,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = torch.cat([enc_output1, enc_output2], dim=-1)

        return enc_output



class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out




