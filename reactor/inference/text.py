# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for the Reactor runtime.
"""Upstream T5 encoder with memory-mapped checkpoint loading."""
import logging

import torch
from worldplay2.text.encoder import T5EncoderModel as NativeT5EncoderModel, umt5_xxl
from worldplay2.text.tokenizer import HuggingfaceTokenizer


class T5EncoderModel(NativeT5EncoderModel):
    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device).eval().requires_grad_(False)
        logging.info(f'loading {checkpoint_path}')
        # Memory-mapped: the checkpoint's pages are file-backed and copied into
        # the parameters on ``device``; the state is never held in anonymous
        # host memory.
        model.load_state_dict(torch.load(
            checkpoint_path, map_location='cpu', mmap=True, weights_only=True))
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace')
