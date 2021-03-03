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
# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

from typing import Set, Optional, List
from nltk.corpus.reader import Synset
from nltk.corpus import wordnet as wn

# import nltk
# nltk.download('wordnet')


def antonyms(sense: str) -> Set[Synset]:
  """
  :param sense: the ID of the sense (e.g., 'dog.n.01').
  :return: a set of Synsets representing the union of all antonyms of the sense as well as its synonyms.
  """
  result = set()

  synset = wn.synset(sense)

  for lemma in synset.lemmas():

    for antonym in lemma.antonyms():
      syn_antonym = antonym.synset()
      result.add(syn_antonym)

      for lemma in syn_antonym.lemmas():
        result.add(lemma.synset())

  return result


def paths(sense_0: str, sense_1: str) -> List[List[Synset]]:
    result = list()

    synset_0 = wn.synset(sense_0)
    synset_1 = wn.synset(sense_1)

    hypernym_paths_0 = synset_0.hypernym_paths()
    hypernym_paths_1 = synset_1.hypernym_paths()

    lowest_common_hypernyms = synset_0.lowest_common_hypernyms(synset_1)

    for lch in lowest_common_hypernyms:
      filtered_paths_0 = filter(lambda path: lch in path, hypernym_paths_0)
      filtered_paths_1 = filter(lambda path: lch in path, hypernym_paths_1)

      mapped_paths_0 = map(lambda path: path[path.index(lch):], filtered_paths_0)
      mapped_paths_1 = map(lambda path: path[path.index(lch):], filtered_paths_1)

      cleaned_paths_0 = map(list, set(map(tuple, mapped_paths_0)))
      cleaned_paths_1 = map(list, set(map(tuple, mapped_paths_1)))

      for path_0 in cleaned_paths_0:

        for path_1 in cleaned_paths_1:
          final_path = path_0[::-1] + path_1[1:]
          result.append(final_path)

    return result or list(list())


if __name__ == '__main__':
    print(antonyms('purchase.v.01'))

    for path in paths('dog.n.01', 'cat.n.01'):
        print([s.name() for s in path])
