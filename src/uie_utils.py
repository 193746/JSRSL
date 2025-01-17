import torch
import numpy as np
import json
import jsonlines

# map from word to (sub)token
def get_tokens_map(tokens):
    i=0
    tokens_map=[]
    for token in tokens:
        tokens_map.append(i)
        if "@@" not in token:
            i+=1
    return tokens_map

def span_word2token(start_list,end_list,tokens_map):
    new_start_list,new_end_list=[],[]
    for i in range(len(start_list)):
        tokens_len=tokens_map.count(i)
        token_start_list = [0 for _ in range(tokens_len)]
        token_end_list = [0 for _ in range(tokens_len)]
        if start_list[i] != 0:
            token_start_list[0] = start_list[i]
        if end_list[i] != 0:
            token_end_list[-1] = end_list[i]
        new_start_list.extend(token_start_list)
        new_end_list.extend(token_end_list)
    return new_start_list,new_end_list

def span_token2word(start_list,end_list,tokens_map):
    new_start_list,new_end_list=[],[]
    for k in range(tokens_map[-1]+1):
        new_start_list.append(start_list[tokens_map.index(k)])
        new_end_list.append(end_list[len(tokens_map)-1-tokens_map[::-1].index(k)])
    return new_start_list,new_end_list

def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    UIE: https://arxiv.org/abs/2203.12277

    Get idx of the last dimension in probability arrays, which is greater than a limitation.

    Args:
        probs (list[list[int]]): The input probability arrays.
        limit (float): The limitation for probability.
        return_prob (bool): Whether to return the probability
    Returns:
        List[int]: The index of the last dimension meet the conditions.
    """
    if any(isinstance(i, list) for i in probs):
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p >= limit:
                if return_prob:
                    result.append((i, round(p.item(),6)))
                else:
                    result.append(i)
        return result

def get_span(start_ids, end_ids, with_prob=False):
    """
    UIE: https://arxiv.org/abs/2203.12277

    Get span set from position start and end list.

    Args:
        start_ids (List[int]/List[tuple]): The start index list.
        end_ids (List[int]/List[tuple]): The end index list.
        with_prob (bool): If True, each element for start_ids and end_ids is a tuple aslike: (index, probability).
    Returns:
        set: The span set without overlapping, every id can only be used once .
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue
        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue
        if start_id > end_id:
            end_pointer += 1
            continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result

# ner evaluation
def get_evaluate_fpr(seqs_hat,seqs_true,start,end,pred_start,pred_end,index,is_test=False):
    index=index.squeeze(dim=-1)

    pred_start_ids_list = get_bool_ids_greater_than(pred_start,limit=1,return_prob=False)
    pred_end_ids_list = get_bool_ids_greater_than(pred_end,limit=1,return_prob=False)

    start_ids_list = get_bool_ids_greater_than(start,limit=1,return_prob=False)
    end_ids_list = get_bool_ids_greater_than(end,limit=1,return_prob=False)

    pred_list = []
    true_list = []
    epoch=len(start)  
   
    for i in range(epoch):
        pred_span_list = get_span(pred_start_ids_list[i], pred_end_ids_list[i], with_prob=False)
        span_list = get_span(start_ids_list[i], end_ids_list[i], with_prob=False)
        for psp in pred_span_list:
            start_index = psp[0]
            end_index = psp[1]
            pred=[index[i].item(),pred_end[i][end_index],seqs_hat[i][start_index:end_index+1]]
            pred_list.append(pred)
        for sp in span_list:
            start_index = sp[0]
            end_index = sp[1]
            true=[index[i].item(),end[i][end_index],seqs_true[i][start_index:end_index+1]]
            true_list.append(true)
    
    if not is_test:
        return get_fpr(pred_list,true_list)
    else:
        return pred_list,true_list

# classify task evaluation
def get_evaluate_fpr_sa(action_prob,action,scenario_prob,scenario,index,is_test=False):
    index=index.squeeze(dim=-1)
    
    true_action=action.squeeze(dim=-1)
    pred_action=action_prob.argmax(dim=-1)

    true_scenario=scenario.squeeze(dim=-1)
    pred_scenario=scenario_prob.argmax(dim=-1)

    seq_num=len(true_action)

    true_list=[[index[i].item(),true_action[i].item(),true_scenario[i].item()] for i in range(seq_num)]
    pred_list=[[index[i].item(),pred_action[i].item(),pred_scenario[i].item()] for i in range(seq_num)]

    if not is_test:
        return get_fpr(pred_list,true_list)
    else:
        return pred_list,true_list

def get_fpr(pred_list,true_list):
    X = len([x for x in pred_list if x in true_list])
    Y = len(pred_list)
    Z = len(true_list)
    
    eps=1e-8
    f1, precision, recall = 2*X/(Y+Z+eps), X/(Y+eps), X/(Z+eps)
    return f1, precision, recall
