# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim):
        super().__init__()

        self.linear1 = nn.Linear(dim, 2*dim)
        self.linear2 = nn.Linear(dim*2, dim)
        self.linear3 = nn.Linear(dim, 1)

    def forward(self, data):
        out1 = F.relu(self.linear1(data))
        out2 = self.linear2(out1)
        out3 = self.linear3(out2)
        out = F.softplus(out3)

        return out


class SpatiotemporalModel(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, event_times, spatial_events, input_mask, t0, t1):
        """
        Args:
            event_times: (N, T)
            spatial_events: (N, T, D)
            input_mask: (N, T)
            t0: () or (N,)
            t1: () or (N,)
        """
        pass


class CombinedSpatiotemporalModel(SpatiotemporalModel):

    def __init__(self, temporal_model, encoder_model=None):
        super().__init__()
        self.encoder = encoder_model
        self.temporal_model = temporal_model
        self.pred = Predictor(dim=1+self.temporal_model.hdim)

    def forward(self, event_times, spatial_events, input_mask, t0, t1):
        if self.encoder:
            spatial_events = self.encoder(spatial_events, event_times, input_mask)
        time_emb, lamda, loglik, pre_hiddens = self._temporal_logprob(event_times, spatial_events, input_mask, t0, t1)
        pre_label = self.pred(torch.cat([time_emb, lamda.unsqueeze(-1)], dim=1))
        return pre_label, loglik, time_emb, pre_hiddens

    def _temporal_logprob(self, event_times, spatial_events, input_mask, t0, t1):
        return self.temporal_model.logprob(event_times, spatial_events, input_mask, t0, t1)


class CombinedSpatiotemporalModel2(SpatiotemporalModel):

    def __init__(self, temporal_model, encoder_model=None):
        super().__init__()
        self.encoder = encoder_model
        self.temporal_model = temporal_model
        self.pred = Predictor(dim=self.encoder.d_model)

    def forward(self, event_times, spatial_events, input_mask, t0, t1):
        if self.encoder:
            spatial_events = self.encoder(spatial_events, event_times, input_mask)
        pre_label = self.pred(spatial_events[:, -1, :])
        return pre_label, torch.zeros(pre_label.shape).to(pre_label)


def zero_diffeq(t, h):
    return torch.zeros_like(h)


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)

