import logging
import os
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import random
import numpy as np
import torch.nn.functional as F

from funasr.layers.abs_normalize import AbsNormalize
from funasr.losses.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from funasr.models.ctc import CTC
from funasr.models.decoder.abs_decoder import AbsDecoder
from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.postencoder.abs_postencoder import AbsPostEncoder
from funasr.models.predictor.cif import mae_loss
from funasr.models.preencoder.abs_preencoder import AbsPreEncoder
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.modules.add_sos_eos import add_sos_eos
from funasr.modules.nets_utils import make_pad_mask, pad_list
from funasr.modules.nets_utils import th_accuracy
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.models.base_model import FunASRModel
from funasr.models.predictor.cif import CifPredictorV3

# from funasr.models.e2e_asr_common import ErrorCalculator
from uie_utils import get_evaluate_fpr,get_evaluate_fpr_sa
from error_calculator import ErrorCalculator
from log_utils import *

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class Paraformer(FunASRModel):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
            self,
            vocab_size: int,
            token_list: Union[Tuple[str, ...], List[str]],
            frontend: Optional[AbsFrontend],
            specaug: Optional[AbsSpecAug],
            normalize: Optional[AbsNormalize],
            encoder: AbsEncoder,
            decoder: AbsDecoder,
            ctc: CTC,
            ctc_weight: float = 0.5,
            interctc_weight: float = 0.0,
            ignore_id: int = -1,
            blank_id: int = 0,
            sos: int = 1,
            eos: int = 2,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
            report_cer: bool = True,
            report_wer: bool = True,
            sym_space: str = "<space>",
            sym_blank: str = "<blank>",
            extract_feats_in_collect_stats: bool = True,
            predictor=None,
            predictor_weight: float = 0.0,
            predictor_bias: int = 0,
            sampling_ratio: float = 0.2,
            share_embedding: bool = False,
            preencoder: Optional[AbsPreEncoder] = None,
            postencoder: Optional[AbsPostEncoder] = None,
            use_1st_decoder_loss: bool = False,
            span_alpha: int=0.5,
            name_entity_num: int=3,
            action_num: int=0,
            scenario_num: int=0,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = blank_id
        self.sos = vocab_size - 1 if sos is None else sos
        self.eos = vocab_size - 1 if eos is None else eos
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        self.name_entity_num=name_entity_num

        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.error_calculator = None

        if ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = decoder

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.predictor = predictor
        self.predictor_weight = predictor_weight
        self.predictor_bias = predictor_bias
        self.sampling_ratio = sampling_ratio
        self.criterion_pre = mae_loss(normalize_length=length_normalized_loss)
        self.step_cur = 0

        self.share_embedding = share_embedding
        if self.share_embedding:
            self.decoder.embed = None

        self.use_1st_decoder_loss = use_1st_decoder_loss

        self.span_alpha=span_alpha

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            index:torch.Tensor,
            index_lengths:torch.Tensor,
            start: torch.Tensor,
            start_lengths: torch.Tensor,
            end: torch.Tensor,
            end_lengths: torch.Tensor,
            action: torch.Tensor=None,
            action_lengths: torch.Tensor=None,
            scenario: torch.Tensor=None,
            scenario_lengths: torch.Tensor=None,
            decoding_ind: int = None,
            is_test:bool=False,
            output_path:str=None,
            wav_dict:Dict[str,str]=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
                decoding_ind: int
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]
        self.step_cur += 1
        # for data-parallel
        text=text[:, : text_lengths.max()]
        speech=speech[:, :speech_lengths.max()]
        start=start[:,:start_lengths.max()]
        end=end[:,:end_lengths.max()]

        # 1. Encoder
        if hasattr(self.encoder, "overlap_chunk_cls"):
            ind = self.encoder.overlap_chunk_cls.random_choice(self.training, decoding_ind)
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, ind=ind)
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, pre_loss_att, acc_att, cer_att, wer_att = None, None, None, None, None
        loss_pre = None
        loss_span = None
        loss = None
        fpr_span = None
        stats = dict()

        if not is_test:
            loss_att,acc_att,cer_att,wer_att,loss_pre,pre_loss_att,loss_span,fpr_span = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths,start,start_lengths,end,end_lengths,action,scenario,index
            )
            loss = loss_att + loss_pre * self.predictor_weight
            if self.use_1st_decoder_loss and pre_loss_att is not None:
                loss = loss +  pre_loss_att
            loss = (1-self.span_alpha) * loss + self.span_alpha * loss_span

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach()
            stats["pre_loss_att"] = pre_loss_att.detach() if pre_loss_att is not None else None
            stats["loss_pre"] = loss_pre.detach()
            stats["loss_span"] = loss_span.detach()   
            stats["loss"] = torch.clone(loss.detach())

            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

            stats["f1"]=fpr_span[0]
            stats["precision"]= fpr_span[1]
            stats["recall"]=fpr_span[2]
            stats["ic_f1"]=fpr_span[3] if fpr_span[3] is not None else None

        else:
            self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths,start,start_lengths,end,end_lengths,action,scenario,index,is_test=is_test,output_path=output_path,wav_dict=wav_dict
            )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), speech.device)
        return loss, stats, weight

    def encode(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor, ind: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            if hasattr(self.encoder, "overlap_chunk_cls"):
                encoder_out, encoder_out_lens, _ = self.encoder(
                    feats, feats_lengths, ctc=self.ctc, ind=ind
                )
                encoder_out, encoder_out_lens = self.encoder.overlap_chunk_cls.remove_chunk(encoder_out,
                                                                                            encoder_out_lens,
                                                                                            chunk_outs=None)
            else:
                encoder_out, encoder_out_lens, _ = self.encoder(
                    feats, feats_lengths, ctc=self.ctc
                )
        else:
            if hasattr(self.encoder, "overlap_chunk_cls"):
                encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths, ind=ind)
                encoder_out, encoder_out_lens = self.encoder.overlap_chunk_cls.remove_chunk(encoder_out,
                                                                                            encoder_out_lens,
                                                                                            chunk_outs=None)
            else:
                encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def _extract_feats(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]
        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_att_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
            start: torch.Tensor,
            start_lens: torch.Tensor,
            end: torch.Tensor,
            end_lens: torch.Tensor,
            action: torch.Tensor,
            scenario: torch.Tensor,
            index:torch.Tensor,
            is_test:bool=False,
            output_path:str=None,
            wav_dict:Dict[str,str]=None,
    ):
        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
            encoder_out.device)
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias
            start = torch.where(start==-1, 0, start)
            start = torch.cat((start,torch.zeros(len(start[:,0]),1).to(start.device)),dim=1)
            start_lens = start_lens + self.predictor_bias
            end = torch.where(end==-1, 0, end)
            end = torch.cat((end,torch.zeros(len(end[:,0]),1).to(end.device)),dim=1)
            end_lens = end_lens + self.predictor_bias

        if not is_test:
            pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(encoder_out, ys_pad, encoder_out_mask,
                                                                                    ignore_id=self.ignore_id)
        else:
            pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(encoder_out, None, encoder_out_mask,
                                                                                    ignore_id=self.ignore_id)


        # 0. sampler
        decoder_out_1st = None
        pre_loss_att = None
        if self.training:
            if self.sampling_ratio > 0.0:
                if self.step_cur < 2:
                    logging.info("enable sampler in paraformer, sampling_ratio: {}".format(self.sampling_ratio))
                if self.use_1st_decoder_loss:
                    sematic_embeds, decoder_out_1st, pre_loss_att = self.sampler_with_grad(encoder_out, encoder_out_lens,
                                                                                        ys_pad, ys_pad_lens,
                                                                                        pre_acoustic_embeds)
                else:
                    sematic_embeds, decoder_out_1st = self.sampler(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,
                                                                pre_acoustic_embeds)
            else:
                if self.step_cur < 2:
                    logging.info("disable sampler in paraformer, sampling_ratio: {}".format(self.sampling_ratio))
                sematic_embeds = pre_acoustic_embeds
        else:
            # logging.info("do evaluating, disable sampler in paraformer")
            sematic_embeds = pre_acoustic_embeds

        if not is_test:
            decoder_out,_,start_prob,end_prob,action_prob,scenario_prob = self.decoder(
                    encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens
                )
        else:
            decoder_out,_,start_prob,end_prob,action_prob,scenario_prob = self.decoder(
                encoder_out, encoder_out_lens, sematic_embeds,pre_token_length
            )

        if not is_test:
            if decoder_out_1st is None:
                decoder_out_1st = decoder_out
            # 2. Compute attention loss
            loss_att = self.criterion_att(decoder_out, ys_pad)

            acc_att = th_accuracy(
                decoder_out_1st.view(-1, self.vocab_size),
                ys_pad,
                ignore_label=self.ignore_id,
            )
            loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)

            # Compute cer/wer using attention-decoder
            if not self.training:
                cer_att, wer_att = None, None
                fpr_span=[None, None, None,None]
            else:
                ys_hat=decoder_out_1st.argmax(dim=-1)
                cer_att,wer_att=self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
                seqs_hat,seqs_true,tokens_maps_hat,tokens_maps_true=self.error_calculator.convert_to_char_list(ys_hat.cpu(), ys_pad.cpu())
                true_start_list,true_end_list=self.error_calculator.convert_to_word_span(start.cpu(),end.cpu(),tokens_maps_true)
                pred_start=torch.where(start_prob>=0.5,1,0) 
                pred_end=end_prob.argmax(dim=-1)
                pred_start_list,pred_end_list=self.error_calculator.convert_to_word_span(pred_start.cpu(),pred_end.cpu(),tokens_maps_hat)
                f1,precision,recall=get_evaluate_fpr(seqs_hat, seqs_true,true_start_list,true_end_list,pred_start_list,pred_end_list,index.cpu(),is_test)
                if torch.is_tensor(action_prob) and torch.is_tensor(scenario_prob):
                    ic_f1,_,_ =get_evaluate_fpr_sa(action_prob.cpu(), action.cpu(),scenario_prob.cpu(),scenario.cpu(),index.cpu(),is_test)
                    fpr_span=[f1, precision, recall, ic_f1]
                else:
                    fpr_span=[f1, precision, recall, None]
        
            # ner loss
            start_ids = start.type(torch.float32)
            end_ids = end.type(torch.long)
            loss_start = torch.nn.functional.binary_cross_entropy(start_prob, start_ids)
            _end_prob = end_prob.view(-1,self.name_entity_num+1)
            _end_ids = end_ids.view(-1)
            loss_end =  torch.nn.functional.cross_entropy(_end_prob, _end_ids)
            loss_ner = loss_start + loss_end

            if torch.is_tensor(action_prob) and torch.is_tensor(scenario_prob):
                # ic loss
                action = action.squeeze(dim=-1)
                action_ids=action.type(torch.long)
                loss_action =  torch.nn.functional.cross_entropy(action_prob, action_ids)
                scenario = scenario.squeeze(dim=-1)
                scenario_ids=scenario.type(torch.long)
                loss_scenario =  torch.nn.functional.cross_entropy(scenario_prob, scenario_ids)
                loss_ic=loss_action+loss_scenario
                loss_span=loss_ner+loss_ic
            else:
                loss_span=loss_ner

            return loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att,loss_span,fpr_span
    
        else:
            ys_hat=decoder_out.argmax(dim=-1)
            seqs_hat,seqs_true,tokens_maps_hat,tokens_maps_true=self.error_calculator.convert_to_char_list_for_test(ys_hat.cpu(), ys_pad.cpu(),pre_token_length.cpu(),ys_pad_lens.cpu())
            true_start_list,true_end_list=self.error_calculator.convert_to_word_span(start.cpu(),end.cpu(),tokens_maps_true)
            pred_start=torch.where(start_prob>=0.5,1,0)
            pred_end=end_prob.argmax(dim=-1)
            pred_start_list,pred_end_list=self.error_calculator.convert_to_word_span(pred_start.cpu(),pred_end.cpu(),tokens_maps_hat)
            ner_pred_list,ner_true_list=get_evaluate_fpr(seqs_hat, seqs_true,true_start_list,true_end_list,pred_start_list,pred_end_list,index.cpu(),is_test)
            if torch.is_tensor(action_prob) and torch.is_tensor(scenario_prob):
                ic_pred_list,ic_true_list =get_evaluate_fpr_sa(action_prob.cpu(),action.cpu(),scenario_prob.cpu(),scenario.cpu(),index.cpu(),is_test)
            else:
                ic_pred_list=None
                ic_true_list=None
            write_transcription([" ".join(seq) for seq in seqs_hat],[" ".join(seq) for seq in seqs_true],os.path.join(output_path,"transcript.txt"))
            write_structure_information(ner_pred_list,ner_true_list,ic_pred_list,ic_true_list,os.path.join(output_path,"structure.jsonl"),wav_dict)
       
    def sampler(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens,pre_acoustic_embeds):

        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        if self.share_embedding:
            ys_pad_embed = self.decoder.output_layer.weight[ys_pad_masked]
        else:
            ys_pad_embed = self.decoder.embed(ys_pad_masked)
        with torch.no_grad():
            decoder_out,_,_,_,_,_ = self.decoder(
                encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens
            )
            # decoder_out, _ = decoder_outs[0], decoder_outs[1]
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
                if target_num > 0:
                    input_mask[li].scatter_(dim=0, index=torch.randperm(seq_lens[li])[:target_num].cuda(), value=0)
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0)
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask

    def sampler_with_grad(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds):
        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        if self.share_embedding:
            ys_pad_embed = self.decoder.output_layer.weight[ys_pad_masked]
        else:
            ys_pad_embed = self.decoder.embed(ys_pad_masked)
        decoder_out,_,_,_,_,_ = self.decoder(
            encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens
        )
        pre_loss_att = self.criterion_att(decoder_out, ys_pad)
        # decoder_out, _ = decoder_outs[0], decoder_outs[1]
        pred_tokens = decoder_out.argmax(-1)
        nonpad_positions = ys_pad.ne(self.ignore_id)
        seq_lens = (nonpad_positions).sum(1)
        same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
        input_mask = torch.ones_like(nonpad_positions)
        bsz, seq_len = ys_pad.size()
        for li in range(bsz):
            target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
            if target_num > 0:
                input_mask[li].scatter_(dim=0, index=torch.randperm(seq_lens[li])[:target_num].cuda(), value=0)
        input_mask = input_mask.eq(1)
        input_mask = input_mask.masked_fill(~nonpad_positions, False)
        input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0)

        return sematic_embeds * tgt_mask, decoder_out * tgt_mask, pre_loss_att

    def _calc_ctc_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

