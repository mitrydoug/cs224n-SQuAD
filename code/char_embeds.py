"""This file contains a function to create default character embeddings
and return them as an embedding matrix. These embeddings are untrained and must
be trained by the model. """

import numpy as np

# Set of characters found in the dev and train sets
all_characters = ['\x83', '\x87', '\x8b', '\x8f', '\x93', '\x97', '\x9b', '\x9f', '\xa3', '$', '\xa7', '(', '\xab', ',', '\xaf', '0', '\xb3', '4', '\xb7', '8', '\xbb', '<', '\xbf', '@', '\xc3', '\xc7', '\xcb', '\xcf', '\xd7', '\xdb', '`', '\xe3', 'd', '\xe7', 'h', '\xeb', 'l', '\xef', 'p', 't', 'x', '|', '\x80', '\x84', '\x88', '\x8c', '\x90', '\x94', '\x98', '\x9c', '\xa0', '#', '\xa4', "'", '\xa8', '+', '\xac', '/', '\xb0', '3', '\xb4', '7', '\xb8', ';', '\xbc', '?', '\xc4', '\xc8', '\xcc', '\xd0', '\xd8', '[', '\xdc', '_', '\xe0', 'c', '\xe4', 'g', '\xe8', 'k', '\xec', 'o', '\xf0', 's', 'w', '{', '\x81', '\x85', '\x89', '\x8d', '\x91', '\x95', '\x99', '\x9d', '\xa1', '"', '\xa5', '&', '\xa9', '*', '\xad', '.', '\xb1', '2', '\xb5', '6', '\xb9', ':', '\xbd', '>', '\xc5', '\xc9', '\xcd', '\xd1', '\xd5', '\xd9', '^', '\xe1', 'b', '\xe5', 'f', '\xe9', 'j', '\xed', 'n', 'r', 'v', 'z', '~', '\x82', '\x86', '\x8a', '\x8e', '\x92', '\x96', '\x9a', '\x9e', '!', '\xa2', '%', '\xa6', ')', '\xaa', '-', '\xae', '1', '\xb2', '5', '\xb6', '9', '\xba', '=', '\xbe', '\xc2', '\xc6', '\xca', '\xce', '\xd2', '\xd6', '\xda', ']', 'a', '\xe2', 'e', '\xe6', 'i', '\xea', 'm', 'q', 'u', 'y', '}']

_PADCHAR = b"<pad>"
_UNKCHAR = b"<unk>"
_START_CHARS = [_PADCHAR, _UNKCHAR]
PAD_CHAR_ID = 0
UNK_CHAR_ID = 1

def get_char_embeddings(char_dim, char_set=all_characters, random_init=True):
    """Returns character embeddings matrix and mappings from characters
    to character ids (i.e. index in the embedding matrix).

    Input:
        char_dim: a command-line flag that determines char embedding dimension
        random_init: if True, init char embeddings with randomness. Otherwise,
            init char embeddings with zeros. Defaults to true.

    Returns:
        char_emb_matrix: Numpy array shape (176, char_dim) containing char
            embeddings, plus PAD and UNK embeddings in the first two rows
        char2id: dictionary mapping char (string) to char id (int)
        id2char: dictionary mapping char id (int) to char (string)
    """
    print "Creating char embeddings"

    num_chars = len(char_set)
    char_emb_matrix = np.zeros((num_chars + len(_START_CHARS), char_dim))
    char2id = {}
    id2char = {}

    char_emb_matrix[:len(_START_CHARS), :] = np.random.rand(len(_START_CHARS), char_dim)

    idx = 0

    # Put PAD and UNK chars in the dictionaries
    for char in _START_CHARS:
        char2id[char] = idx
        id2char[idx] = char
        idx += 1

    # Create embeddings for all other characters
    for char in char_set:
        if random_init:
            vector = np.random.rand(char_dim)
        else:
            vector = np.zeros(char_dim)
        char_emb_matrix[idx, :] = vector
        char2id[char] = idx
        id2char[idx] = char
        idx += 1

    return char_emb_matrix, char2id, id2char

def test_char_embeddings():
    chars = ['a', 'b', 'c', 'd']
    dim = 3
    matrix, char2id, id2char = get_char_embeddings(dim, chars)
    assert matrix.shape == (len(chars) + len(_START_CHARS), dim)
    print char2id
    print {
        _PADCHAR: 0,
        _UNKCHAR: 1,
        'a': 2,
        'b': 3,
        'c': 4,
        'd': 5
    }
    assert cmp(char2id, {
        _PADCHAR: 0,
        _UNKCHAR: 1,
        'a': 2,
        'b': 3,
        'c': 4,
        'd': 5
    }) == 0
    assert cmp(id2char, {
        0: _PADCHAR,
        1: _UNKCHAR,
        2: 'a',
        3: 'b',
        4: 'c',
        5: 'd'
    }) == 0
    print "All tests passed!"

if __name__ == '__main__':
    test_char_embeddings()