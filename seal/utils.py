# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


def _remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def load_state_dict_from_lightning_checkpoint(model, path):
    state_dict = torch.load(path, map_location="cpu")
    # state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    # for key in ['shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']:
    #         state_dict[key] = torch.cat([state_dict[key], torch.zeros_like(state_dict[key][:1])], 0)
    # _remove_ignore_keys_(state_dict)
    # if hasattr(model, "lm_head"):
    #     model.lm_head = _make_linear_from_emb(model.model.shared)
    model.load_state_dict(state_dict)


def load_state_dict_from_fairseq_checkpoint(model, path):
    state_dict = torch.load(path, map_location="cpu")["model"]
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    for key in ["shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]:
        state_dict[key] = torch.cat([state_dict[key], torch.zeros_like(state_dict[key][:1])], 0)
    _remove_ignore_keys_(state_dict)
    if hasattr(model, "lm_head"):
        model.lm_head = _make_linear_from_emb(model.model.shared)
    model.model.load_state_dict(state_dict)
