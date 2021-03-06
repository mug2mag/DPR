#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
import logging
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn
import pdb

from dpr.data.biencoder_data import BiEncoderSample
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import CheckpointState

logger = logging.getLogger(__name__)

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "question_ids",
        "question_segments",
        "context_ids",
        "ctx_segments",
        "is_positive",
        "hard_negatives",
        "encoder_type",
        "questions",
        "positive_contexts",
        "negative_contexts",
        "all_contexts",
    ],
)
# TODO: it is only used by _select_span_with_token. Move them to utils
rnd = random.Random(0)


def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    print('ctx_vectors.shape', ctx_vectors.shape)
    r = torch.matmul(torch.transpose(q_vectors, 1, 2), ctx_vectors)
    return r


from transformers.modeling_bert import BertEncoder, BertConfig, BertModel

class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        config,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,

    ):
        self.config = config
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder
        trans_layer = nn.TransformerEncoderLayer(768, 12)  # 12???head?????????
        self.trans_encoder = nn.TransformerEncoder(trans_layer, 1)
        cfg = BertConfig.from_pretrained("bert-base-uncased")
        # cfg = BertConfig.from_pretrained("/data/home/scv2223/archive/bert-base-uncased", local_files_only=True)

        cfg.num_hidden_layers = 1
        # self.encoder = BertEncoder(cfg)
        self.encoder = BertModel(cfg)
        self.dense = nn.Linear(768 * 2, 768)
        self.final_dense = nn.Linear(768, 1)

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
    ) -> (T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                )

        return sequence_output, pooled_output, hidden_states

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos=0,
    ) -> Tuple[T, T]:
        # question_ids ??? context_ids ????????????cls???sep???
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
        )
        # todo: ??????question_segments???ctx_segments??????????????????

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
            ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder
        )  # q_pooled_out: 8 x 768  ctx_pooled_out: 16 x 768

        extended_question_ids = self.double_question(question_ids)

        extended_q_seq = self.double_question(_q_seq)

        # ctx_attn_mask = ctx_attn_mask.unsqueeze(-1)
        extended_question_attn_mask = self.double_question(question_attn_mask)
        # last_question_attn_mask = last_question_attn_mask.unsqueeze(-1)

        # """
       # ??????????????? position embeddings ??? type embeddings
        q_token_type_ids = torch.zeros_like(extended_question_ids)
        c_token_type_ids = torch.ones_like(context_ids)
        x_type_ids = torch.cat([q_token_type_ids, c_token_type_ids], dim=1).long()
        x_attn_mask = torch.cat([extended_question_attn_mask, ctx_attn_mask], dim=1)
       # """

        x = torch.cat([extended_q_seq, _ctx_seq], dim=1)

        encoder_outputs = self.encoder(inputs_embeds=x, token_type_ids=x_type_ids, attention_mask=x_attn_mask)
        sequence_output = encoder_outputs[0]

        avg_pooled = sequence_output.mean(1)
        max_pooled = torch.max(sequence_output, dim=1)
        pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)
        pooled = self.dense(pooled)

        predictions = self.final_dense(pooled)

        return predictions

    def double_question(self, quesitons):
        """
        ?????????????????????????????????context(?????????)??????????????????????????????????????????????????????????????????????????????context
        """
        cat_result = []
        for single_q in quesitons:
            single_cat_q = torch.cat([single_q.unsqueeze(0)]*2)
            cat_result.append(single_cat_q)
        cat_result = torch.cat(cat_result, dim=0)
        return cat_result

    @classmethod
    def create_biencoder_input2(
        cls,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []
        questions = []
        positive_contexts = []
        negative_contexts = []
        all_contexts = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            positive_contexts.append(positive_ctx.text)
            all_contexts.append(positive_ctx.text)

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages

            # question = normalize_question(sample.query)

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]
            negative_contexts.append(hard_neg_ctxs[0].text)
            all_contexts.append(hard_neg_ctxs[0].text)

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            question = sample.query
            questions.append(question)
            # question = cls.build_question_with_special_tokens(question)  # ??????cls??????

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(
                    ctx.text, title=ctx.title if (insert_title and ctx.title) else None
                )
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )   # ?????????????????????negative?????????, ???????????????????????????????????????????????????all_ctxs??????????????????negative????????????

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(
                        question, tensorizer, token_str=query_token
                    )
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(
                        tensorizer.text_to_tensor(" ".join([query_token, question]))
                    )
            else:
                question_tensors.append(tensorizer.text_to_tensor(question)) # ????????????????????????cxt??????????????? ???dl4macro???????????????

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0) # ctx ????????????256???ctxs_tensor???(32, 256), ????????????  (batch_size???16?????????
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)  # (16, 256)  (batch_size???16

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)  # ?????????????????????????????????ones_like. ?????????????????????????????????????????????????????????

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
            questions,
            positive_contexts,
            negative_contexts,
            all_contexts
        )

    @classmethod
    def build_question_with_special_tokens(
            cls, input
    ) -> str:
        if isinstance(input, str):
            return '[CLS]' + input + '[SEP]'
        if isinstance(input, list):
            context_list = [ctx.text + ctx.title for ctx in input]
            return '[CLS]'+''.join(context_list) + '[SEP]'

    def load_state(self, saved_state: CheckpointState):
        # TODO: make a long term HF compatibility fix
        if "question_model.embeddings.position_ids" in saved_state.model_dict:
            del saved_state.model_dict["question_model.embeddings.position_ids"]
            del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        self.load_state_dict(saved_state.model_dict, strict=False)

    def get_state_dict(self):
        return self.state_dict()


class BiEncoderNllLoss(object):
    def calc(
        self,
        input,
        linear_output: T,
        # ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        # hard_negative_idx_per_question??????????????????
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        # scores = self.get_scores(q_vectors, ctx_vectors)  # ??????????????? q???c???dot  q:16 x 768, c: 32 x 768

        q_num = linear_output.size(0)
        scores = linear_output.view(q_num, -1)  # [4,2]

        #### manual test ??????????????? ###
        # questions = input.questions
        # all_contexts = input.all_contexts
        # negative_contexts = input.negative_contexts
        # positive_contexts = input.positive_contexts
        sigmoid_scores = torch.sigmoid(scores)  # [4, 2]
        #### manual test ###

        prediction_scores = scores.view(-1)
        new_positive_idx_per_question = [float(0)]*q_num
        for i in positive_idx_per_question:
            new_positive_idx_per_question[i] = float(1)

        # loss = F.nll_loss(
        #     softmax_scores,
        #     torch.tensor(new_positive_idx_per_question).to(softmax_scores.device),
        #     reduction="mean",
        # )
        loss_fun = nn.BCEWithLogitsLoss()
        # loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(prediction_scores, torch.tensor(new_positive_idx_per_question).to(scores.device))
        # loss = self.compute_loss(sigmoid_scores, new_positive_idx_per_question)

        correct_predictions_count = [sigmoid_scores[index] > sigmoid_scores[index+1] for index in positive_idx_per_question if index+1 < len(sigmoid_scores)].count(True)

        # correct_predictions_count = torch.tensor([len([i for i in q_vectors if i > 0.5])])

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    def compute_loss(self, predictions, labels):
        # ?????????????????????????????????, ???????????????????????????
        predictions = predictions.view(-1)
        labels = labels.view(-1)
        epsilon = 1e-8
        # ?????????
        loss =\
            - labels * torch.log(predictions + epsilon) - \
            (torch.tensor(1.0) - labels) * torch.log(torch.tensor(1.0) - predictions + epsilon)
        # ?????????, ????????????????????????loss
        # loss???????????????
        loss = torch.mean(loss)
        return loss

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores


def _select_span_with_token(
    text: str, tensorizer: Tensorizer, token_str: str = "[START_ENT]"
) -> T:
    id = tensorizer.get_token_id(token_str)
    query_tensor = tensorizer.text_to_tensor(text)

    if id not in query_tensor:
        query_tensor_full = tensorizer.text_to_tensor(text, apply_max_len=False)
        token_indexes = (query_tensor_full == id).nonzero()
        if token_indexes.size(0) > 0:
            start_pos = token_indexes[0, 0].item()
            # add some randomization to avoid overfitting to a specific token position

            left_shit = int(tensorizer.max_length / 2)
            rnd_shift = int((rnd.random() - 0.5) * left_shit / 2)
            left_shit += rnd_shift

            query_tensor = query_tensor_full[start_pos - left_shit :]
            cls_id = tensorizer.tokenizer.cls_token_id
            if query_tensor[0] != cls_id:
                query_tensor = torch.cat([torch.tensor([cls_id]), query_tensor], dim=0)

            from dpr.models.reader import _pad_to_len

            query_tensor = _pad_to_len(
                query_tensor, tensorizer.get_pad_id(), tensorizer.max_length
            )
            query_tensor[-1] = tensorizer.tokenizer.sep_token_id
            # logger.info('aligned query_tensor %s', query_tensor)

            assert id in query_tensor, "query_tensor={}".format(query_tensor)
            return query_tensor
        else:
            raise RuntimeError(
                "[START_ENT] toke not found for Entity Linking sample query={}".format(
                    text
                )
            )
    else:
        return query_tensor
