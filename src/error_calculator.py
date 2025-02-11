import json
import logging
import sys

from itertools import groupby
import numpy as np
import six

from postprocess_utils import sentence_postprocess,sentence_postprocess2
from uie_utils import *

class ErrorCalculator(object):
    """Calculate CER and WER for E2E_ASR and CTC models during training.

    :param y_hats: numpy array with predicted text
    :param y_pads: numpy array with true (target) text
    :param char_list:
    :param sym_space:
    :param sym_blank:
    :return:
    """

    def __init__(
        self, char_list, sym_space, sym_blank, report_cer=False, report_wer=False
    ):
        """Construct an ErrorCalculator object."""
        super(ErrorCalculator, self).__init__()

        self.report_cer = report_cer
        self.report_wer = report_wer

        self.char_list = char_list
        self.space = sym_space
        self.blank = sym_blank
        self.idx_blank = self.char_list.index(self.blank)
        if self.space in self.char_list:
            self.idx_space = self.char_list.index(self.space)
        else:
            self.idx_space = None

    def __call__(self, ys_hat, ys_pad, is_ctc=False):
        """Calculate sentence-level WER/CER score.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :param bool is_ctc: calculate CER score for CTC
        :return: sentence-level WER score
        :rtype float
        :return: sentence-level CER score
        :rtype float
        """
        cer, wer = None, None
        if is_ctc:
            return self.calculate_cer_ctc(ys_hat, ys_pad)
        elif not self.report_cer and not self.report_wer:
            return cer, wer

        seqs_hat, seqs_true = self.convert_to_char(ys_hat, ys_pad)
        if self.report_cer:
            cer = self.calculate_cer(seqs_hat, seqs_true)

        if self.report_wer:
            wer = self.calculate_wer(seqs_hat, seqs_true)
        return cer, wer

    def calculate_cer_ctc(self, ys_hat, ys_pad):
        """Calculate sentence-level CER score for CTC.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :return: average sentence-level CER score
        :rtype float
        """
        import editdistance

        cers, char_ref_lens = [], []
        for i, y in enumerate(ys_hat):
            y_hat = [x[0] for x in groupby(y)]
            y_true = ys_pad[i]
            seq_hat, seq_true = [], []
            for idx in y_hat:
                idx = int(idx)
                if idx != -1 and idx != self.idx_blank and idx != self.idx_space:
                    seq_hat.append(self.char_list[int(idx)])

            for idx in y_true:
                idx = int(idx)
                if idx != -1 and idx != self.idx_blank and idx != self.idx_space:
                    seq_true.append(self.char_list[int(idx)])

            hyp_chars = "".join(seq_hat)
            ref_chars = "".join(seq_true)
            if len(ref_chars) > 0:
                cers.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))

        cer_ctc = float(sum(cers)) / sum(char_ref_lens) if cers else None
        return cer_ctc

    def convert_to_char(self, ys_hat, ys_pad):
        """Convert index to character.

        :param torch.Tensor seqs_hat: prediction (batch, seqlen)
        :param torch.Tensor seqs_true: reference (batch, seqlen)
        :return: token list of prediction
        :rtype list
        :return: token list of reference
        :rtype list
        """
        seqs_hat, seqs_true = [], []
        for i, y_hat in enumerate(ys_hat):
            y_true = ys_pad[i]
            eos_true = np.where(y_true == -1)[0]
            ymax = eos_true[0] if len(eos_true) > 0 else len(y_true)
            # NOTE: padding index (-1) in y_true is used to pad y_hat
            
            seq_hat = [self.char_list[int(idx)] for idx in y_hat[:ymax]]
            while self.space in seq_hat:
                seq_hat.remove(self.space)
            while self.blank in seq_hat:
                seq_hat.remove(self.blank)
            seq_hat_text=sentence_postprocess2(seq_hat)[0]  # tokens to text

            seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
            while self.space in seq_true:
                seq_true.remove(self.space)
            seq_true_text=sentence_postprocess2(seq_true)[0]
            
            seqs_hat.append(seq_hat_text)
            seqs_true.append(seq_true_text)
        return seqs_hat, seqs_true

    def convert_to_char_list(self, ys_hat, ys_pad):
        seqs_hat, seqs_true = [], []
        tokens_maps_hat,tokens_maps_true=[],[]
        for i, y_hat in enumerate(ys_hat):
            y_true = ys_pad[i]
            eos_true = np.where(y_true == -1)[0]
            ymax = eos_true[0] if len(eos_true) > 0 else len(y_true)
            # NOTE: padding index (-1) in y_true is used to pad y_hat
            
            seq_hat = [self.char_list[int(idx)] for idx in y_hat[:ymax]]
            while self.space in seq_hat:
                seq_hat.remove(self.space)
            while self.blank in seq_hat:
                seq_hat.remove(self.blank)
            tokens_map_hat=get_tokens_map(seq_hat[:-1]) # remove eos
            seq_hat_text=sentence_postprocess2(seq_hat)[1]  

            
            seq_true = [self.char_list[int(idx)] for idx in y_true[:ymax]]
            while self.space in seq_true:
                seq_true.remove(self.space)
            tokens_map_true=get_tokens_map(seq_true[:-1])
            seq_true_text=sentence_postprocess2(seq_true)[1]
            
            seqs_hat.append(seq_hat_text)
            seqs_true.append(seq_true_text)

            tokens_maps_hat.append(tokens_map_hat)
            tokens_maps_true.append(tokens_map_true)
        return seqs_hat, seqs_true, tokens_maps_hat,tokens_maps_true

    def convert_to_char_list_for_test(self, ys_hat, ys_pad,ys_hat_lens,ys_pad_lens):
        seqs_hat, seqs_true = [], []
        tokens_maps_hat,tokens_maps_true=[],[]
        for i, y_hat in enumerate(ys_hat):            
            hat_len=int(ys_hat_lens[i].item())
            seq_hat = [self.char_list[int(idx)] for idx in y_hat[:hat_len]]
            tokens_map_hat=get_tokens_map(seq_hat[:-1]) 
            seq_hat_text=sentence_postprocess2(seq_hat)[1]

            y_true = ys_pad[i]
            pad_len=int(ys_pad_lens[i].item())
            seq_true = [self.char_list[int(idx)] for idx in y_true[:pad_len]]
            tokens_map_true=get_tokens_map(seq_true[:-1])
            seq_true_text=sentence_postprocess2(seq_true)[1]
            
            seqs_hat.append(seq_hat_text)
            seqs_true.append(seq_true_text)

            tokens_maps_hat.append(tokens_map_hat)
            tokens_maps_true.append(tokens_map_true)
        return seqs_hat, seqs_true, tokens_maps_hat,tokens_maps_true
    
    # tensor,tensor,list=>list,list
    def convert_to_word_span(self,tokens_starts,tokens_ends,tokens_maps):
        start_list=[]
        end_list=[]
        for i in range(len(tokens_maps)):
            length=len(tokens_maps[i])
            tokens_map=tokens_maps[i]
            tokens_start=tokens_starts[i][:length].numpy().tolist()
            tokens_end=tokens_ends[i][:length].numpy().tolist()
            if len(tokens_map)==0:
                word_start=tokens_start
                word_end=tokens_end
            else:
                word_start,word_end=span_token2word(tokens_start,tokens_end,tokens_map)
            start_list.append(word_start)
            end_list.append(word_end)
        return start_list,end_list

    def calculate_cer(self, seqs_hat, seqs_true):
        """Calculate sentence-level CER score.

        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level CER score
        :rtype float
        """
        import editdistance

        char_eds, char_ref_lens = [], []
        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            hyp_chars = seq_hat_text.replace(" ", "")
            ref_chars = seq_true_text.replace(" ", "")
            char_eds.append(editdistance.eval(hyp_chars, ref_chars))
            char_ref_lens.append(len(ref_chars))
        return float(sum(char_eds)) / sum(char_ref_lens)

    def calculate_wer(self, seqs_hat, seqs_true):
        """Calculate sentence-level WER score.

        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level WER score
        :rtype float
        """
        import editdistance

        word_eds, word_ref_lens = [], []
        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            hyp_words = seq_hat_text.split()
            ref_words = seq_true_text.split()
            word_eds.append(editdistance.eval(hyp_words, ref_words))
            word_ref_lens.append(len(ref_words))
        return float(sum(word_eds)) / sum(word_ref_lens)
