# ========================================================================
# Copyright 2021 Emory University
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

import re

values = {
  "one": 1,
  "two": 2,
  "three": 3,
  "four": 4,
  "five": 5,
  "six": 6,
  "seven": 7,
  "eight": 8,
  "nine": 9,
  "ten": 10,
  "eleven": 11,
  "twelve": 12,
  "thirteen": 13,
  "fourteen": 14,
  "fifteen": 15,
  "sixteen": 16,
  "seventeen": 17,
  "eighteen": 18,
  "nineteen": 19,
  "twenty": 20,
  "thirty": 30,
  "fourty": 40,
  "fifty": 50,
  "sixty": 60,
  "seventy": 70,
  "eighty": 80,
  "ninety": 90,
  "hundred": 100,
  "thousand": 1000,
  "million": 1000000,
  "billion": 1000000000,
  "trillion": 1000000000000
}

def tokenizer(text):
  raw_tokens = re.split(r"(\W)", text)
  new_tokens = []

  STARTS = ["'"]
  ENDS = [",", ".", "'"]

  def aux(token):
    start = next((s for s in STARTS if token.startswith(s)), None)
    if start:
      n = len(start)
      t1 = token[:n]
      t2 = token[n:]
      new_tokens.append(t1)
      aux(t2)
      return

    end = next((e for e in ENDS if token.endswith(e)), None)
    if end:
      n = len(end)
      t1 = token[:-n]
      t2 = token[-n:]
      aux(t1)
      new_tokens.append(t2)
      return

    new_tokens.append(token)

  for token in raw_tokens:
    aux(token)

  return new_tokens


def convert(number):
  multiple = 0
  result = 0

  for x in number:
    if x in values:
      value = values[x]

      if value % 100:
        multiple += value
      else:
        multiple *= value

        if value != 100:
          result += multiple
          multiple = 0

  return str(result + multiple)


def normalize(text):
  # split tokens at space, period or hyphen
  delimiter = re.compile(r"\W")
  tokens = tokenizer(text)
  length = len(tokens)

  number = []
  output = []
  index = 0

  while index < length:
    token = tokens[index]
    lower = token.lower()
    after = next((x for x in tokens[index + 1:] if not delimiter.match(x)), None)

    if lower in values:
      number.append(lower)

    elif ((lower == "a" or lower == "an") and (after in values)):
      number.append("one")

    elif number and after not in values:
      output.append(convert(number))
      output.append(token)
      number = []

    elif delimiter.match(token):
      if not number:
        output.append(token)

    else:
      output.append(token)

    index += 1

  if number:
    output.append(convert(number))

  print("".join(output))
  return "".join(output)


def normalize_extra(text):
  # TODO: to be updated
  return text


if __name__ == '__main__':
  S = [
    'I met twelve people',
    'I have one brother and two sisters',
    'A year has three hundred sixty five days',
    'I made a million, dollars',
    'I can count to twelve million two-hundred-five thousand six-hundred-thirty-three',
  ]

  T = [
    'I met 12 people',
    'I have 1 brother and 2 sisters',
    'A year has 365 days',
    'I made 1000000, dollars',
    'I can count to 12205633',
  ]

  correct = 0
  for s, t in zip(S, T):
    if normalize(s) == t:
      correct += 1

  print('Score: {}/{}'.format(correct, len(S)))
