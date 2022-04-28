# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import UserDict
from typing import *
import warnings

from more_itertools import chunked
import torch
from torch import nn
import torch.distributed as dist

from transformers import LogitsProcessor, BeamScorer, BeamSearchScorer, LogitsProcessorList, StoppingCriteriaList, HammingDiversityLogitsProcessor
from transformers.generation_utils import BeamSearchOutput, validate_stopping_criteria, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput
from transformers.generation_logits_process import TopKLogitsWarper

from seal.index import FMIndex

stopword_token_ids = [
    10,   # a 
    41,   # an
    660,  # An
    5,    # the
    1941, # THE
    20,   # The
    7,    # to
    6,    # and
]

class IndexBasedLogitsProcessor(LogitsProcessor):
    """
    Class that masks logit, meant to be used during decoding. The logit mask is determined by finding the range of rows
    in the FM-index that correspond to the previously decoded token ( $O(n log V)$ ), then finding all tokens in that
    interval ( $O(V log V)$ ).
    """
    def __init__(
            self, 
            index: FMIndex,
            num_beams: int, 
            pad_token_id: int = 0,
            eos_token_id: int = 2,
            force_decoding_from: Optional[List[int]] = None,
            stop_at_count: int = 0,
            always_allow_eos: bool = False,
            forced_bos_token_id: Optional[int] = None,
        ):
        self.index = index
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self._num_beams = num_beams
        self.log_odds_weight = 0.0
        self.force_decoding_from = force_decoding_from
        self.force_decoding_second_token = None
        self.block_initial_stopwords = False
        self.stop_at_count = stop_at_count
        self.always_allow_eos = always_allow_eos
        self.forced_bos_token_id = forced_bos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        mask = torch.full_like(scores, float('-inf'))

        if self.forced_bos_token_id is not None:
            if input_ids.size(1) == 1:
                mask[:, self.forced_bos_token_id] = 0.0
                return scores + mask
            else:
                input_ids = input_ids[:, 1:]

        if input_ids.size(1) == 1:

            distinct = self.index.occurring_distinct
            distinct = torch.LongTensor(distinct).to(scores.device)
            mask[:, distinct] = 0.0

        else:

            input_ids_list = input_ids.view(-1, self._num_beams, input_ids.shape[-1]).tolist()  

            lows = []
            highs = []
            fm_index_counts = []

            for batch_id, beam_sent in enumerate(input_ids_list):
                for beam_id, sent in enumerate(beam_sent):

                    if sent[-1] in (self.eos_token_id, self.pad_token_id):
                        low = 0
                        high = 0
                        count = 0

                    elif self.force_decoding_from is not None:
                        low, high = self.index.get_range(self.force_decoding_from + sent[1:])
                        count = self.index.get_count(self.force_decoding_from + sent[1:-1])

                    else:
                        low, high = self.index.get_range(sent[1:])
                        count = self.index.get_count(sent[1:-1])

                    lows.append(low)
                    highs.append(high)
                    fm_index_counts.append(count)

            fm_index_result = self.index.get_distinct_count_multi(lows, highs)
            fm_index_result = fm_index_result[::-1]
            fm_index_counts = fm_index_counts[::-1]

            for batch_id, beam_sent in enumerate(input_ids_list):
                for beam_id, sent in enumerate(beam_sent):

                    if self.stop_at_count > 0 and fm_index_counts[-1] <= self.stop_at_count:
                        fm_index_result.pop()
                        fm_index_counts.pop()
                        distinct = [self.eos_token_id]

                    elif sent[-1] == self.eos_token_id:
                        fm_index_result.pop()
                        fm_index_counts.pop()
                        distinct = [self.pad_token_id]

                    elif sent[-1] == self.pad_token_id:
                        fm_index_result.pop()
                        fm_index_counts.pop()
                        distinct = [self.pad_token_id]

                    else:
                        fm_index_counts.pop()
                        distinct, _ = fm_index_result.pop()

                    distinct = torch.LongTensor(distinct).to(scores.device)

                    mask[batch_id * self._num_beams + beam_id, distinct] = 0

        if self.always_allow_eos:
            mask[:, self.eos_token_id] = 0.0

        return scores + mask


def constrained_beam_search(
        model,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        constrained_decoding_processor: Optional[IndexBasedLogitsProcessor] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        sample: bool = False,
        topk: int = 0,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:

        if topk > 0:
            topk_warper = TopKLogitsWarper(topk)

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
        output_scores = output_scores if output_scores is not None else model.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            if topk:
                next_token_logits = topk_warper(input_ids, next_token_logits)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)

            next_token_scores_no_prev = next_token_scores_processed
            next_token_scores = next_token_scores_no_prev + beam_scores[:, None].expand_as(next_token_scores)

            if constrained_decoding_processor is not None:
                next_token_scores_constrained_no_prev = constrained_decoding_processor(input_ids, next_token_scores_processed)
                next_token_scores_constrained = next_token_scores_constrained_no_prev + beam_scores[:, None].expand_as(next_token_scores)
                # if return_masked_scores:
                #     next_token_scores = next_token_scores_constrained
            else:
                next_token_scores_constrained_no_prev = next_token_scores_no_prev
                next_token_scores_constrained = next_token_scores

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            
            if sample:
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
                weights = next_token_scores_constrained_no_prev.view(batch_size * num_beams, vocab_size).log_softmax(-1).exp()
                nans = torch.isnan(weights.sum(-1))
                weights[nans, :] = 0.0
                weights[nans, eos_token_id] = 1.0
                next_tokens = torch.multinomial(weights, 1, replacement=True).view(batch_size, 1 * num_beams)
                next_token_scores = next_token_scores.gather(-1, next_tokens)
                # next_token_scores = next_token_scores.reshape(batch_size, num_beams, 1)
                # next_token_scores[:, :, :] = 0.0
                # next_token_scores = next_token_scores.reshape(batch_size, 1 * num_beams)
            else:
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
                next_token_scores_constrained = next_token_scores_constrained.view(batch_size, num_beams * vocab_size)
                next_token_scores_constrained, next_tokens = torch.topk(
                    next_token_scores_constrained, 2 * num_beams, dim=1, largest=True, sorted=True
                )
                next_token_scores = next_token_scores.gather(-1, next_tokens)

            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = model._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            else:
                num_return_sequences = beam_scorer.num_beam_hyps_to_keep
                # return only as many indices as sequences
                beam_indices = tuple(
                    (beam_indices[i * num_beams : i * num_beams + num_return_sequences] for i in range(batch_size))
                )
                beam_indices = sum(beam_indices, ())

            if model.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

@torch.inference_mode()
def fm_index_generate(
    model,
    index: FMIndex,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    min_length: int = 3,
    max_length: int = 25,
    length_penalty: float = 1.0,
    num_beams: int = 3,
    diverse_bs_groups: int = 1,
    diverse_bs_penalty: float = 0.0,
    eos_token_id: Optional[int] = None,
    force_decoding_from: Optional[List[int]] = None,
    always_allow_eos: bool = False,
    keep_history: bool = False,
    disable_fm_index: bool = False,
    sample: bool = False,
    stop_at_count: int = 0,
    topk: int = 0,
    transformers_output: bool = False,
    **kwargs,
    ):

    if 'forced_bos_token_id' in kwargs:
        forced_bos_token_id = kwargs.pop('forced_bos_token_id')
    else:
        forced_bos_token_id = model.config.forced_bos_token_id
    
    if sample:
        orig_num_beams = num_beams
        input_ids = input_ids.repeat(num_beams, 1)
        attention_mask = attention_mask.repeat(num_beams, 1)
        num_beams = 1

    device = input_ids.device
    if eos_token_id is None:
        eos_token_id = model.config.eos_token_id

    logits_processor = model._get_logits_processor(
        encoder_input_ids=input_ids,
        repetition_penalty=None,
        no_repeat_ngram_size=0,
        encoder_no_repeat_ngram_size=0,
        bad_words_ids=None,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=None,
        prefix_allowed_tokens_fn=None,
        forced_bos_token_id=forced_bos_token_id,
        forced_eos_token_id=None,
        num_beams=num_beams,
        num_beam_groups=1,
        diversity_penalty=0.0,
        remove_invalid_values=True)

    if diverse_bs_groups > 1 and diverse_bs_penalty > 0.0:
        logits_processor.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=diverse_bs_penalty,
                num_beams=num_beams,
                num_beam_groups=diverse_bs_groups,
            )
        )

    
    if not disable_fm_index:
        constrained_decoding_processor = IndexBasedLogitsProcessor(
                num_beams=num_beams // diverse_bs_groups,
                index=index,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=eos_token_id or model.config.eos_token_id,
                force_decoding_from=force_decoding_from,
                stop_at_count=stop_at_count,
                always_allow_eos=always_allow_eos,
                forced_bos_token_id=forced_bos_token_id,
            )
        if diverse_bs_groups > 1:
            logits_processor.append(constrained_decoding_processor)
    else:
        constrained_decoding_processor = None

    stopping_criteria = model._get_stopping_criteria(
        max_length=max_length,
        max_time=None,
        #max_new_tokens=None,
        #start_length=None
        )


    model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
        input_ids, {'attention_mask': attention_mask})
    model_kwargs['use_cache'] = True

    decoder_input_ids = model._prepare_decoder_input_ids_for_generation(
        batch_size=input_ids.size(0),
        decoder_start_token_id=model.config.decoder_start_token_id, 
        bos_token_id=model.config.bos_token_id,
    )
    
    if keep_history:

        beam_scorer = BeamSearchScorerWithMemory(
            batch_size=decoder_input_ids.shape[0],
            num_beams=num_beams,
            device=device,
            length_penalty=length_penalty,
            do_early_stopping=False,
            num_beam_hyps_to_keep=num_beams,
            min_length=min_length,
            max_length=max_length,
            num_beam_groups=diverse_bs_groups,
        )

    else:

        beam_scorer = BeamSearchScorer(
            batch_size=decoder_input_ids.shape[0],
            num_beams=num_beams,
            device=device,
            length_penalty=length_penalty,
            do_early_stopping=False,
            num_beam_hyps_to_keep=num_beams,
            num_beam_groups=diverse_bs_groups,
        )

    decoder_input_ids, model_kwargs = model._expand_inputs_for_generation(
        decoder_input_ids, 
        expand_size=num_beams, 
        is_encoder_decoder=True, 
        **model_kwargs)
    
    if diverse_bs_groups > 1:
        out = model.group_beam_search(
            input_ids=decoder_input_ids,
            beam_scorer=beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            output_scores=True,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=eos_token_id,
            **model_kwargs)
        
    else:
        out = constrained_beam_search(
            model,
            input_ids=decoder_input_ids,
            beam_scorer=beam_scorer,
            logits_processor=logits_processor,
            constrained_decoding_processor=constrained_decoding_processor,
            stopping_criteria=stopping_criteria,
            output_scores=True,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=eos_token_id,
            sample=sample,
            topk=topk,
            **model_kwargs)

    if transformers_output:
        return out

    if sample:
        out = [[(h[0] * h[1].size(0) ** length_penalty, h[1].tolist()) for b in bb for h in b.beams if h[0] > float('-inf')] for bb in chunked(beam_scorer._beam_hyps, orig_num_beams)]
    else:
        out = [[(h[0] * h[1].size(0) ** length_penalty, h[1].tolist()) for h in b.beams if h[0] > float('-inf')] for b in beam_scorer._beam_hyps]

    return out

class BeamSearchScorerWithMemory(BeamScorer):

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        min_length: Optional[int] = 15,
        max_length: Optional[int] = 25,
        **kwargs,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups
        self.min_length = min_length
        self.max_length = max_length

        self._is_init = False
        self._beam_hyps = [
            BeamHypothesesWithMemory(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                min_length=self.min_length,
                max_length=self.max_length)
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)


        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                f"`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` "
                f"has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

        if "max_length" in kwargs:
            warnings.warn(
                "Passing `max_length` to BeamSearchScorer is deprecated and has no effect."
                "`max_length` should be passed directly to `beam_search(...)`, `beam_sample(...)`"
                ",or `group_beam_search(...)`."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0

            broken = False
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                beam_hyp.add(
                    torch.cat([
                        input_ids[batch_beam_idx],
                        next_token.view(1),
                    ]),
                    next_score.item(),
                )
                # add to generated hypotheses if end of sentence
                if broken:
                    pass

                elif (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    pass
                    
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    broken = True

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):

            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(
                    final_tokens.clone(),
                    final_score)

        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, 3)
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
            }
        )

class BeamHypothesesWithMemory:
    
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool, min_length: int, max_length: int):
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9
        self.min_length = min_length
        self.max_length = max_length
        self._best = None

    def __len__(self):
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float):
        size = hyp.size(0)
        score = sum_logprobs / (size ** self.length_penalty)
        self.beams.append((score, hyp))

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        return cur_len >= self.max_length

