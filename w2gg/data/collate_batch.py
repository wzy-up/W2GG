import torch
from w2gg.structures import TLGBatch


class BatchCollator(object):
    """Collect a dataset batch while preserving ragged pseudo-glance lists."""

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        (
            feats,
            queries,
            wordlens,
            gmasks2d,
            moments,
            num_sentence,
            glances,
            glance_weights,
            primary_glances,
            idxs,
        ) = transposed_batch

        return (
            TLGBatch(
                feats=torch.stack(feats).float(),
                queries=queries,
                wordlens=wordlens,
                gmasks2d=gmasks2d,
                moments=moments,
                num_sentence=num_sentence,
                glances=glances,
                glance_weights=glance_weights,
                primary_glances=primary_glances,
            ),
            idxs,
        )
