import copy
import math
import random
from typing import *

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from transformers import modeling_bart as bart


class AMRBartForConditionalGeneration(bart.BartForConditionalGeneration):
    base_model_prefix = "model"

    def __init__(self, config: bart.BartConfig, backpointer_idx=None):
        super().__init__(config)

    def prepare_logits_for_generation(self, logits, cur_len, max_length):
        #if cur_len == 1:
        #    self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits


