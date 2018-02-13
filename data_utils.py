"""Text utilities for creating vocabularies, tokenizing, encoding.
   Insipration from Tensorflow documentation (www.tensorflow.org).
   The data reading functions were taken from
   https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re

import tensorflow as tf

# Special vocabulary symbols - these get placed at the start of vocab.
_PAD = "_PAD"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w.lower() for w in words if w]


def create_vocabulary(vocab_path_stem, data_path, vocabulary_size,
                      tokenizer=None):
  """Create vocabulary files (if they not do exist) from data file.

  Assumes lines in the input data are: head_word gloss_word1
  gloss_word2... The glosses are tokenised, and the gloss vocab
  contains the most frequent tokens up to vocabulary_size. Vocab is
  written to vocabulary_path in a one-token-per-line format, so that
  the token in the first line gets id=0, second line gets id=1, and so
  on. A list of all head words is also written.

  We also assume that the final processed glosses won't contain any
  instances of the corresponding head word (since we don't want a word
  defined in terms of itself), and the vocab counts calculated here
  will reflect that.

  Args:
    vocab_path_stem: path where the vocab files will be created.
    data_path: data file that will be used to create vocabulary.
    vocabulary_size: limit on the size of the vocab (glosses and heads).
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
  """
  # Check if the vocabulary already exists; if so, do nothing.
  if not tf.gfile.Exists(vocab_path_stem + ".vocab"):
    print("Creating vocabulary %s from data %s" % (vocab_path_stem, data_path))
    # Counts for the head words.
    head = collections.defaultdict(int)
    # Counts for all words (heads and glosses).
    words = collections.defaultdict(int)
    with tf.gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing training data line %d" % counter)
        #line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        words[tokens[0]] += 1
        head[tokens[0]] += 1
        for word in tokens[1:]:
          # Assume final gloss won't contain the corresponding head.
          if word != tokens[0]:
            words[word] += 1
      # Sort words by frequency, adding _PAD and _UNK to all_words.
      all_words = _START_VOCAB + sorted(words, key=words.get, reverse=True)
      head_vocab = sorted(head, key=head.get, reverse=True)
      print("Writing out vocabulary")
      assert len(all_words) >= vocabulary_size, (
          "vocab size must be less than %s, the total"
          "no. of words in the training data" % len(all_words))
      # Write the head words to file.
      with tf.gfile.GFile(
          vocab_path_stem + "_all_head_words.txt", mode="w") as head_file:
        for w in head_vocab:
          head_file.write(w + "\n")
      # Write the vocab words to file.
      with tf.gfile.GFile(vocab_path_stem + ".vocab", mode="w") as vocab_file:
        for w in all_words[:vocabulary_size]:
          vocab_file.write(w + "\n")
      print("Data pre-processing complete")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file vocabulary_path.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if tf.gfile.Exists(vocabulary_path):
    rev_vocab = []
    with tf.gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    #rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = {x: y for (y, x) in enumerate(rev_vocab)}
    rev_vocab = {y: x for x, y in vocab.items()}
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None,
                          normalize_digits=True):
  """Convert a string to list of integers representing token ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  else:
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def pad_sequence(sequence, max_seq_len):
  padding_required = max_seq_len - len(sequence)
  # Sentence too long, so truncate.
  if padding_required < 0:
    padded = sequence[:max_seq_len]
  # Sentence too short, so pad.
  else:
    padded = sequence + ([PAD_ID] * padding_required)
  return padded


def data_to_token_ids(data_path, target_path, vocabulary_path, max_seq_len,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  Loads data line-by-line from data_path, calls sentence_to_token_ids,
  and saves the result to target_path. See sentence_to_token_ids on
  the details of token-ids format. Also pads out each sentence with
  the _PAD id, or truncates, so that each sentence is the same length.

  We also remove any instance of the head word from the corresponding
  gloss, since we don't want to define any word in terms of itself.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    max_seq_len: maximum sentence length before truncation applied.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  # Check if token-id version of the data exists; if so, do nothing.
  if not (tf.gfile.Exists(target_path + ".gloss") and
          tf.gfile.Exists(target_path + ".head")):
    print("Encoding data into token-ids in %s" % data_path)
    # vocab is a mapping from tokens to ids.
    vocab, _ = initialize_vocabulary(vocabulary_path + ".vocab")
    # Write each data line to an id-based heads and glosses file.
    with tf.gfile.GFile(data_path, mode="r") as data_file:
      with tf.gfile.GFile(target_path + ".gloss", mode="w") as glosses_file:
        with tf.gfile.GFile(target_path + ".head", mode="w") as heads_file:
          counter = 0
          for line in data_file:
            counter += 1
            if counter % 100000 == 0:
              print("encoding training data line %d" % counter)
            token_ids = sentence_to_token_ids(
              line, vocab, tokenizer, normalize_digits)
              #tf.compat.as_bytes(line), vocab, tokenizer, normalize_digits)
            # Write out the head ids, one head per line.
            heads_file.write(str(token_ids[0]) + "\n")
            # Remove all instances of head word in gloss.
            clean_gloss = [w for w in token_ids[1:] if w != token_ids[0]]
            # Pad out the glosses, or truncate, so all the same length.
            glosses_ids = pad_sequence(clean_gloss, max_seq_len)
            # Write out the glosses as ids, one gloss per line.
            glosses_file.write(" ".join([str(t) for t in glosses_ids]) + "\n")


def prepare_dict_data(data_dir, train_file, dev_file,
                      vocabulary_size, max_seq_len, tokenizer=None):
  """Get processed data into data_dir, create vocabulary.

  Args:
    data_dir: directory in which the data sets will be stored.
    train_file: file with dictionary definitions for training.
    dev_file: file with dictionary definitions for development testing.
    vocabulary_size: size of the vocabulary to create and use.
    max_seq_len: maximum sentence length before applying truncation.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
  """
  train_path = os.path.join(data_dir, train_file)
  dev_path = os.path.join(data_dir, dev_file)

  # Create vocabulary of the appropriate size.
  vocab_path_stem = os.path.join(data_dir, "definitions_%d" % vocabulary_size)
  create_vocabulary(vocab_path_stem, train_path, vocabulary_size, tokenizer)

  # Create versions of the train and dev data with token ids.
  train_ids = os.path.join(
      data_dir, "train.definitions.ids%d" % vocabulary_size)
  dev_ids = os.path.join(data_dir, "dev.definitions.ids%d" % vocabulary_size)
  data_to_token_ids(
      train_path, train_ids, vocab_path_stem, max_seq_len, tokenizer)
  data_to_token_ids(
      dev_path, dev_ids, vocab_path_stem, max_seq_len, tokenizer)
