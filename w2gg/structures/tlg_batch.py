from dataclasses import dataclass
import torch


@dataclass
class TLGBatch(object):
    feats: torch.Tensor
    queries: list
    wordlens: list
    gmasks2d: list
    moments: list
    num_sentence: list
    glances: list
    glance_weights: list
    primary_glances: list

    def to(self, device):
        self.feats = self.feats.to(device)
        self.queries = [query.to(device) for query in self.queries]
        self.wordlens = [word_len.to(device) for word_len in self.wordlens]
        self.gmasks2d = [gmask2d.to(device) for gmask2d in self.gmasks2d]
        self.moments = [moment.to(device) for moment in self.moments]
        self.glances = [
            [g.to(device) for g in glance_per_video]
            for glance_per_video in self.glances
        ]
        self.glance_weights = [
            [w.to(device) for w in weights_per_video]
            for weights_per_video in self.glance_weights
        ]
        self.primary_glances = [g.to(device) for g in self.primary_glances]
        return self
