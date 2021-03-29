# ========================================================================
# Copyright 2020 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import pickle
from collections import Counter
from typing import List, Tuple, Dict, Any

DUMMY = '!@#$'

def read_data(filename: str):
    data, sentence = [], []
    fin = open(filename)

    for line in fin:
        l = line.split()
        if l:
            sentence.append((l[0], l[1]))
        else:
            data.append(sentence)
            sentence = []

    return data


def to_probs(model: Dict[Any, Counter]) -> Dict[str, List[Tuple[str, float]]]:
    probs = dict()
    for feature, counter in model.items():
        ts = counter.most_common()
        total = sum([count for _, count in ts])
        probs[feature] = [(label, count/total) for label, count in ts]
    return probs

def count_to_probs(model):
    total = sum(model.values())
    return {x: (count/total) for x, count in model.items()}


def evaluate(data: List[List[Tuple[str, str]]], *args):
    total, correct = 0, 0
    for sentence in data:
        tokens, gold = tuple(zip(*sentence))
        pred = [t[0] for t in predict(tokens, *args)]
        total += len(tokens)
        correct += len([1 for g, p in zip(gold, pred) if g == p])
    accuracy = 100.0 * correct / total
    return accuracy


def create_dictionaries(data):
    cw_dict = dict()    # key: curr word
    pp_dict = dict()    # key: prev pos
    pw_dict = dict()    # key: prev word
    nw_dict = dict()    # key: next word
    pnw_dict = dict()   # key: (prev word, next word)
    cnw_dict = dict()   # key: (curr word, next word)
    pcw_dict = dict()   # key: (prev word, curr word)
    pcnw_dict = dict()  # key: (prev word, curr word, next word)

    ic_count = Counter() # pos distribution for words where first letter is capitalized
    au_count = Counter() # pos distribution for words where all letters are uppercase
    al_count = Counter() # pos distribution for words where all letters are lowercase
    if_count = Counter() # pos distribution for words where word is first in sentence
    il_count = Counter() # pos distribution for words where word is last in sentence
    hh_count = Counter() # pos distribution for words where word contains a hyphen

    for sentence in data:
        for i, (curr_word, curr_pos) in enumerate(sentence):
            prev_pos = sentence[i-1][1] if i > 0 else DUMMY
            prev_word = sentence[i-1][0] if i else DUMMY
            next_word = sentence[i+1][0] if i+1 < len(sentence) else DUMMY

            cw_dict.setdefault(curr_word, Counter()).update([curr_pos])
            pp_dict.setdefault(prev_pos, Counter()).update([curr_pos])
            pw_dict.setdefault(prev_word, Counter()).update([curr_pos])
            nw_dict.setdefault(next_word, Counter()).update([curr_pos])
            pnw_dict.setdefault((prev_word, next_word), Counter()).update([curr_pos])
            cnw_dict.setdefault((curr_word, next_word), Counter()).update([curr_pos])
            pcw_dict.setdefault((prev_word, curr_word), Counter()).update([curr_pos])
            pcnw_dict.setdefault((prev_word, curr_word, next_word), Counter()).update([curr_pos])

            if curr_word[0] != curr_word[0].lower():        ic_count.update([curr_pos])
            if curr_word == curr_word.upper():              au_count.update([curr_pos])
            if curr_word == curr_word.lower():              al_count.update([curr_pos])
            if curr_word == sentence[0][0]:                 if_count.update([curr_pos])
            if curr_word == sentence[len(sentence)-1][0]:   il_count.update([curr_pos])
            if '-' in curr_word:                            hh_count.update([curr_pos])

    return (
        to_probs(cw_dict),
        to_probs(pp_dict),
        to_probs(pw_dict),
        to_probs(nw_dict),
        to_probs(pnw_dict),
        to_probs(cnw_dict),
        to_probs(pcw_dict),
        to_probs(pcnw_dict),
        count_to_probs(ic_count),
        count_to_probs(au_count),
        count_to_probs(al_count),
        count_to_probs(if_count),
        count_to_probs(il_count),
        count_to_probs(hh_count),
    )


def train(trn_data: List[List[Tuple[str, str]]], dev_data: List[List[Tuple[str, str]]]) -> Tuple:
    """
    :param trn_data: the training set
    :param dev_data: the development set
    :return: a tuple of all parameters necessary to perform part-of-speech tagging
    """

    cw_dict, pp_dict, pw_dict, nw_dict, pnw_dict, cnw_dict, pcw_dict, pcnw_dict, ic_count, au_count, al_count, if_count, il_count, hh_count = create_dictionaries(trn_data)
    best_acc, best_args = -1, None
    grid = [0.1, 0.5, 1.0]

    # manually entered weights after approximation through sub-grid searches of max O(n^5)

    for cw_weight in [1.0]:
        for pp_weight in [0.5]:
            for pw_weight in [0.5]:
                for nw_weight in [0.5]:
                    for pnw_weight in [0.5]:
                        for cnw_weight in [1.0]:
                            for pcw_weight in [1.0]:
                                for pcnw_weight in [1.0]:
                                    for ic_weight in [0.1]:
                                        for au_weight in [1.0]:
                                            for al_weight in [0.1]:
                                                for if_weight in [0.1]:
                                                    for il_weight in [0.1]:
                                                        for hh_weight in [1.0]:
                                                            args = (cw_dict, pp_dict, pw_dict, nw_dict, pnw_dict, cnw_dict, pcw_dict, pcnw_dict, ic_count, au_count, al_count, if_count, il_count, hh_count, cw_weight, pp_weight, pw_weight, nw_weight, pnw_weight, cnw_weight, pcw_weight, pcnw_weight, ic_weight, au_weight, al_weight, if_weight, il_weight, hh_weight)
                                                            acc = evaluate(dev_data, *args)
                                                            print('{:5.2f}% - cw: {:3.1f}, pp: {:3.1f}, pw: {:3.1f}, nw: {:3.1f}, pnw: {:3.1f}, cnw: {:3.1f}, pcw: {:3.1f}, pcnw: {:3.1f}, ic: {:3.1f}, au: {:3.1f}, al: {:3.1f}, if: {:3.1f}, il: {:3.1f}, hh: {:3.1f}'.format(acc, cw_weight, pp_weight, pw_weight, nw_weight, pnw_weight, cnw_weight, pcw_weight, pcnw_weight, ic_weight, au_weight, al_weight, if_weight, il_weight, hh_weight))
                                                            if acc > best_acc: best_acc, best_args = acc, args

    return best_args


def predict(tokens: List[str], *args) -> List[Tuple[str, float]]:
    cw_dict, pp_dict, pw_dict, nw_dict, pnw_dict, cnw_dict, pcw_dict, pcnw_dict, ic_count, au_count, al_count, if_count, il_count, hh_count, cw_weight, pp_weight, pw_weight, nw_weight, pnw_weight, cnw_weight, pcw_weight, pcnw_weight, ic_weight, au_weight, al_weight, if_weight, il_weight, hh_weight = args
    output = []

    for i in range(len(tokens)):
        scores = dict()
        curr_word = tokens[i]
        prev_pos = output[i-1][0] if i > 0 else DUMMY
        prev_word = tokens[i-1] if i > 0 else DUMMY
        next_word = tokens[i+1] if i+1 < len(tokens) else DUMMY

        for pos, prob in cw_dict.get(curr_word, list()):
            scores[pos] = scores.get(pos, 0) + prob * cw_weight

        for pos, prob in pp_dict.get(prev_pos, list()):
            scores[pos] = scores.get(pos, 0) + prob * pp_weight

        for pos, prob in pw_dict.get(prev_word, list()):
            scores[pos] = scores.get(pos, 0) + prob * pw_weight

        for pos, prob in nw_dict.get(next_word, list()):
            scores[pos] = scores.get(pos, 0) + prob * nw_weight

        for pos, prob in pnw_dict.get((prev_word, next_word), list()):
            scores[pos] = scores.get(pos, 0) + prob * pnw_weight

        for pos, prob in cnw_dict.get((curr_word, next_word), list()):
            scores[pos] = scores.get(pos, 0) + prob * cnw_weight

        for pos, prob in pcw_dict.get((prev_word, curr_word), list()):
            scores[pos] = scores.get(pos, 0) + prob * pcw_weight

        for pos, prob in pcnw_dict.get((prev_word, curr_word, next_word), list()):
            scores[pos] = scores.get(pos, 0) + prob * pcnw_weight

        for pos, prob in ic_count.items():
            if curr_word[0] == curr_word[0].upper():
                scores[pos] = scores.get(pos, 0) + prob * ic_weight

        for pos, prob in au_count.items():
            if curr_word == curr_word.upper():
                scores[pos] = scores.get(pos, 0) + prob * au_weight

        for pos, prob in al_count.items():
            if curr_word == curr_word.lower():
                scores[pos] = scores.get(pos, 0) + prob * al_weight

        for pos, prob in if_count.items():
            if i == 0:
                scores[pos] = scores.get(pos, 0) + prob * if_weight

        for pos, prob in il_count.items():
            if i == len(tokens)-1:
                scores[pos] = scores.get(pos, 0) + prob * il_weight

        for pos, prob in hh_count.items():
            if '-' in curr_word:
                scores[pos] = scores.get(pos, 0) + prob * hh_weight

        o = max(scores.items(), key=lambda t: t[1]) if scores else ('XX', 0.0)
        output.append(o)

    return output


if __name__ == '__main__':
    path = './../../'  # path to the cs329 directory
    trn_data = read_data(path + 'dat/pos/wsj-pos.trn.gold.tsv')
    dev_data = read_data(path + 'dat/pos/wsj-pos.dev.gold.tsv')
    model_path = path + 'src/quiz/quiz3.pkl'

    # save model
    # args = train(trn_data, dev_data)
    # pickle.dump(args, open(model_path, 'wb'))

    # load model
    args = pickle.load(open(model_path, 'rb'))
    print(evaluate(dev_data, *args))