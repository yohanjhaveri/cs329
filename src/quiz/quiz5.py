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
import glob
import os
from types import SimpleNamespace
from typing import Iterable, Tuple, Any, List, Set

import ahocorasick


def create_ac(data: Iterable[Tuple[str, Any]]) -> ahocorasick.Automaton:
    """
    Creates the Aho-Corasick automation and adds all (span, value) pairs in the data and finalizes this matcher.
    :param data: a collection of (span, value) pairs.
    """
    AC = ahocorasick.Automaton(ahocorasick.STORE_ANY)

    for span, value in data:
        if span in AC:
            t = AC.get(span)
        else:
            t = SimpleNamespace(span=span, values=set())
            AC.add_word(span, t)
        t.values.add(value)

    AC.make_automaton()
    return AC


def read_gazetteers(dirname: str) -> ahocorasick.Automaton:
    data = []
    for filename in glob.glob(os.path.join(dirname, '*.txt')):
        label = os.path.basename(filename)[:-4]
        for line in open(filename):
            data.append((line.strip(), label))
    return create_ac(data)


def match(AC: ahocorasick.Automaton, tokens: List[str]) -> List[Tuple[str, int, int, Set[str]]]:
    """
    :param AC: the finalized Aho-Corasick automation.
    :param tokens: the list of input tokens.
    :return: a list of tuples where each tuple consists of
             - span: str,
             - start token index (inclusive): int
             - end token index (exclusive): int
             - a set of values for the span: Set[str]
    """
    smap, emap, idx = dict(), dict(), 0
    for i, token in enumerate(tokens):
        smap[idx] = i
        idx += len(token)
        emap[idx] = i
        idx += 1

    # find matches
    text = ' '.join(tokens)
    spans = []
    for eidx, t in AC.iter(text):
        eidx += 1
        sidx = eidx - len(t.span)
        sidx = smap.get(sidx, None)
        eidx = emap.get(eidx, None)
        if sidx is None or eidx is None: continue
        spans.append((t.span, sidx, eidx + 1, t.values))

    return spans


def remove_overlaps(entities: List[Tuple[str, int, int, Set[str]]]) -> List[Tuple[str, int, int, Set[str]]]:
    """
    :param entities: a list of tuples where each tuple consists of
             - span: str,
             - start token index (inclusive): int
             - end token index (exclusive): int
             - a set of values for the span: Set[str]
    :return: a list of entities where each entity is represented by a tuple of (span, start index, end index, value set)
    """
    entities.sort(key=lambda x: x[2])
    sublists = []

    def aux(end, sub):
        for entity in entities:
            s = entity[1]
            e = entity[2]

            if end <= s:
                aux(e, sub + [entity])
            else:
                sublists.append(sub)

    aux(-1, [])

    keys = {(len(sublist), sum([entity[2] - entity[1] for entity in sublist])): sublist for sublist in sublists}
    key = sorted(keys.keys())[-1]

    return keys[key]


def to_bilou(tokens: List[str], entities: List[Tuple[str, int, int, str]]) -> List[str]:
    """
    :param tokens: a list of tokens.
    :param entities: a list of tuples where each tuple consists of
             - span: str,
             - start token index (inclusive): int
             - end token index (exclusive): int
             - a named entity tag
    :return: a list of named entity tags in the BILOU notation with respect to the tokens
    """
    tags = ["O" for token in tokens]

    for entity in entities:
        s = entity[1]
        e = entity[2]
        t = entity[3]

        if (e - s) == 1:
            tags[s] = "U-" + t

        else:
            for i in range(s, e):
                if i == s:          tags[i] = "B-" + t
                elif i == e - 1:    tags[i] = "L-" + t
                else:               tags[i] = "I-" + t

    return tags


if __name__ == '__main__':
    gaz_dir = './../../dat/ner'
    AC = read_gazetteers(gaz_dir)

    tokens = 'Atlantic City of Georgia'.split()
    entities = match(AC, tokens)
    entities = remove_overlaps(entities)
    print(entities)

    # tokens = 'South Korea United States'.split()
    # entities = match(AC, tokens)
    # entities = remove_overlaps(entities)
    # # entities = remove_overlaps([('D', 0, 1, {'country'}), ('B', 1, 2, {'country'}), ('A', 2, 3, {'country'}), ('B A C', 1, 4, {'country'})])
    # # entities = remove_overlaps([('Atlantic City', 0, 2, {'country'}), ('City of Georgia', 1, 3, {'country'})])
    # # entities = remove_overlaps([('South Korea', 0, 2, {'country'}), ('Korea United', 1, 3, {'country'}), ('United States', 2, 4, {'country'})])
    # print(entities)
    # tokens = 'Jinho is a professor at Emory University in the United States of America'.split()
    # entities = [
    #     ('Jinho', 0, 1, 'PER'),
    #     ('Emory University', 5, 7, 'ORG'),
    #     ('United States of America', 9, 13, 'LOC')
    # ]
    # tags = to_bilou(tokens, entities)
    # for token, tag in zip(tokens, tags): print(token, tag)