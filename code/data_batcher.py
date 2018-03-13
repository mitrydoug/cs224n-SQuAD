# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains code to read tokenized data from file,
truncate, pad and process it into batches ready for training"""

from __future__ import absolute_import
from __future__ import division

import random
import time
import re

import numpy as np
from six.moves import xrange
from vocab import PAD_ID, UNK_ID
from char_embeds import PAD_CHAR_ID, UNK_CHAR_ID


class Batch(object):
    """A class to hold the information needed for a training batch"""

    def __init__(self, context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens, qn_char_ids,
                 context_char_ids, ans_span, ans_tokens, uuids=None):
        """
        Inputs:
          {context/qn}_ids: Numpy arrays.
            Shape (batch_size, {context_len/question_len}). Contains padding.
          {context/qn}_mask: Numpy arrays, same shape as _ids.
            Contains 1s where there is real data, 0s where there is padding.
          {context/qn/ans}_tokens: Lists length batch_size, containing lists (unpadded) of tokens (strings)
          ans_span: numpy array, shape (batch_size, 2)
          uuid: a list (length batch_size) of strings.
            Not needed for training. Used by official_eval mode.
        """
        self.context_ids = context_ids
        self.context_mask = context_mask
        self.context_tokens = context_tokens

        self.qn_ids = qn_ids
        self.qn_mask = qn_mask
        self.qn_tokens = qn_tokens

        self.qn_char_ids = qn_char_ids
        self.context_char_ids = context_char_ids

        self.ans_span = ans_span
        self.ans_tokens = ans_tokens

        self.uuids = uuids

        self.batch_size = len(self.context_tokens)


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def intstr_to_intlist(string):
    """Given a string e.g. '311 9 1334 635 6192 56 639', returns as a list of integers"""
    return [int(s) for s in string.split()]


def sentence_to_token_ids(sentence, word2id):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    tokens = split_by_whitespace(sentence) # list of strings
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    return tokens, ids

def token_to_char_ids(token, char2id, max_word_len):
    """Turns a word into a list of character ids, padded to the desired length. The input token should
    already be truncated to the max_word_len, and padding is added up to max_word_len"""
    return [char2id.get(ch, UNK_CHAR_ID) for ch in token] + [PAD_CHAR_ID] * (max_word_len - len(token))


def tokens_to_char_ids(tokens, char2id, max_word_len):
    """
    Turns a list of tokens into a 2D list of character ids.
    :param tokens: list of tokens, has shape ({question/context}_len)
    :param char2id: maps characters to character ids
    :param max_word_len: maximum word length. Tokens of length greater than max_word_len are truncated, and words
    of length less than max_word_length are padded with PAD_CHAR_ID
    :return: 2D padded list of character ids, has shape ({question/context}_len, max_word_len).
    """
    return [token_to_char_ids(word[:max_word_len], char2id, max_word_len) for word in tokens]


def padded_char_ids(batch_word_ids, id2word, char2id, max_word_len):
    """
    Turns a 2D list of token ids into a 3D list of character ids
    :param batch_word_ids: padded list of word ids of shape (batch_size, {question/context}_len)
    :param id2word: maps word ids to words
    :param char2id: maps characters to character ids
    :param max_word_len: the maximum length of a word.
    :return: 3D list containing character ids, has shape (batch_size, {question/context}_len, max_word_len). When a word
    id is either PAD_ID or UNK_ID, it is extended into [PAD_CHAR_ID] * max_word_len
    """
    batch_tokens = [[id2word[word_id] if word_id not in (PAD_ID, UNK_ID) else '' for word_id in id_list] for id_list in
                    batch_word_ids]  # Converts list of ids into list of words, replacing padding with empty string
    return [tokens_to_char_ids(token_list, char2id, max_word_len) for token_list in batch_tokens]


def padded(token_batch, batch_pad=0):
    """
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to. If 0, pad to maximum length sequence in token_batch.
    Returns:
      List (length batch_size) of padded of lists of ints.
        All are same length - batch_pad if batch_pad!=0, otherwise the maximum length in token_batch
    """
    maxlen = max(map(lambda x: len(x), token_batch)) if batch_pad == 0 else batch_pad
    return map(lambda token_list: token_list + [PAD_ID] * (maxlen - len(token_list)), token_batch)

def refill_batches(batches, word2id, context_file, qn_file, ans_file, batch_size, context_len, question_len, word_len, discard_long):
    """
    Adds more batches into the "batches" list.

    Inputs:
      batches: list to add batches to
      word2id: dictionary mapping word (string) to word id (int)
      context_file, qn_file, ans_file: paths to {train/dev}.{context/question/answer} data files
      batch_size: int. how big to make the batches
      context_len, question_len: max length of context and question respectively
      discard_long: If True, discard any examples that are longer than context_len or question_len.
        If False, truncate those exmaples instead.
    """
    print "Refilling batches..."
    tic = time.time()
    examples = [] # list of (qn_ids, context_ids, ans_span, ans_tokens) triples
    context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline() # read the next line from each

    while context_line and qn_line and ans_line: # while you haven't reached the end

        # Convert tokens to word ids
        context_tokens, context_ids = sentence_to_token_ids(context_line, word2id)
        qn_tokens, qn_ids = sentence_to_token_ids(qn_line, word2id)
        ans_span = intstr_to_intlist(ans_line)

        # read the next line from each file
        context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()

        # get ans_tokens from ans_span
        assert len(ans_span) == 2
        if ans_span[1] < ans_span[0]:
            print "Found an ill-formed gold span: start=%i end=%i" % (ans_span[0], ans_span[1])
            continue
        ans_tokens = context_tokens[ans_span[0] : ans_span[1]+1] # list of strings

        # discard or truncate too-long questions
        if len(qn_ids) > question_len:
            if discard_long:
                continue
            else: # truncate
                qn_ids = qn_ids[:question_len]

        # discard or truncate too-long contexts
        if len(context_ids) > context_len:
            if discard_long:
                continue
            else: # truncate
                context_ids = context_ids[:context_len]

        # add to examples
        examples.append((context_ids, context_tokens, qn_ids, qn_tokens, ans_span, ans_tokens))

        # stop refilling if you have 160 batches
        if len(examples) == batch_size * 160:
            break

    # Once you've either got 160 batches or you've reached end of file:

    # Sort by question length
    # Note: if you sort by context length, then you'll have batches which contain the same context many times (because each context appears several times, with different questions)
    examples = sorted(examples, key=lambda e: len(e[2]))

    # Make into batches and append to the list batches
    for batch_start in xrange(0, len(examples), batch_size):

        # Note: each of these is a list length batch_size of lists of ints (except on last iter when it might be less than batch_size)
        context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch = zip(*examples[batch_start:batch_start+batch_size])

        batches.append((context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch))

    # shuffle the batches
    random.shuffle(batches)

    toc = time.time()
    print "Refilling batches took %.2f seconds" % (toc-tic)
    return


def get_batch_generator(word2id, id2word, char2id, context_path, qn_path, ans_path, batch_size, context_len, question_len, word_len, discard_long):
    """
    This function returns a generator object that yields batches.
    The last batch in the dataset will be a partial batch.
    Read this to understand generators and the yield keyword in Python: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do

    Inputs:
      word2id: dictionary mapping word (string) to word id (int)
      context_file, qn_file, ans_file: paths to {train/dev}.{context/question/answer} data files
      batch_size: int. how big to make the batches
      context_len, question_len: max length of context and question respectively
      discard_long: If True, discard any examples that are longer than context_len or question_len.
        If False, truncate those exmaples instead.
    """
    context_file, qn_file, ans_file = open(context_path), open(qn_path), open(ans_path)
    batches = []

    while True:
        if len(batches) == 0: # add more batches
            refill_batches(batches, word2id, context_file, qn_file, ans_file, batch_size, context_len, question_len, word_len, discard_long)
        if len(batches) == 0:
            break

        # Get next batch. These are all lists length batch_size
        (context_ids, context_tokens, qn_ids, qn_tokens, ans_span, ans_tokens) = batches.pop(0)

        # Pad context_ids and qn_ids
        qn_ids = padded(qn_ids, question_len) # pad questions to length question_len
        context_ids = padded(context_ids, context_len) # pad contexts to length context_len

        # Create context_char_ids and qn_char_ids
        qn_char_ids = padded_char_ids(qn_ids, id2word, char2id, word_len)
        context_char_ids = padded_char_ids(context_ids, id2word, char2id, word_len)

        # Make qn_ids into a np array and create qn_mask
        qn_ids = np.array(qn_ids) # shape (question_len, batch_size)
        qn_mask = (qn_ids != PAD_ID).astype(np.int32) # shape (question_len, batch_size)

        # Make context_ids into a np array and create context_mask
        context_ids = np.array(context_ids) # shape (context_len, batch_size)
        context_mask = (context_ids != PAD_ID).astype(np.int32) # shape (context_len, batch_size)

        # Make {qn/context}_char_ids into np arrays
        qn_char_ids = np.array(qn_char_ids)
        context_char_ids = np.array(context_char_ids)

        # Make ans_span into a np array
        ans_span = np.array(ans_span) # shape (batch_size, 2)

        # Make into a Batch object
        batch = Batch(context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens, qn_char_ids, context_char_ids, ans_span, ans_tokens)

        yield batch

    return


def test_padded_char_ids():
    batch = [[4, 2, PAD_ID], [4, UNK_ID, 3]]
    id2word = {4: 'up', 2: 'down', 3: 'h'}
    char2id = {'u': 2, 'p': 3, 'd': 4, 'o': 5, 'h': 6, 'i': 7}
    expected_output = [[[2, 3], [4, 5], [PAD_CHAR_ID, PAD_CHAR_ID]], [[2, 3], [PAD_CHAR_ID, PAD_CHAR_ID], [6, PAD_CHAR_ID]]]
    output = padded_char_ids(batch, id2word, char2id, 2)
    assert expected_output == output
    print 'All tests passed!'

if __name__ == '__main__':
    test_padded_char_ids()
