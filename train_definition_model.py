"""Language encoder for dictionary definitions.

For training, takes (target-word, dictionary-definition) pairs and
optimises the encoder to produce a single vector for each definition
which is close to the vector for the corresponding target word.

The definitions encoder can be either a bag-of-words or an RNN model.

The vectors for the target words, and the words making up the
definitions, can be either pre-trained or learned as part of the
training process.

Sometimes the definitions are referred to as "glosses", and the target
words as "heads".

Inspiration from Tensorflow documentation (www.tensorflow.org).
The data reading functions were taken from
https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os
import sys

import numpy as np
import scipy.spatial.distance as dist
import tensorflow as tf

import data_utils

tf.app.flags.DEFINE_integer("max_seq_len", 20, "Maximum length (in words) of a"
                            "definition processed by the model")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.0001,
                          "Learning rate applied in TF optimiser")
tf.app.flags.DEFINE_integer("embedding_size", 500,
                            "Number of units in word representation.")
tf.app.flags.DEFINE_integer("vocab_size", 100000, "Nunber of words the model"
                            "knows and stores representations for")
tf.app.flags.DEFINE_integer("num_epochs", 1000, "Train for this number of"
                            "sweeps through the training set")
tf.app.flags.DEFINE_string("data_dir", "../data/definitions/", "Directory for finding"
                           "training data and dumping processed data.")
tf.app.flags.DEFINE_string("train_file", "train.definitions.ids100000",
                           "File with dictionary definitions for training.")
tf.app.flags.DEFINE_string("dev_file", "'dev.definitions.ids100000",
                           "File with dictionary definitions for dev testing.")
tf.app.flags.DEFINE_string("save_dir", "/tmp/", "Directory for saving model."
                           "If using restore=True, directory to restore from.")
tf.app.flags.DEFINE_boolean("restore", False, "Restore a trained model"
                            "instead of training one.")
tf.app.flags.DEFINE_boolean("evaluate", False, "Evaluate model (needs" 
                            "Restore==True).")
tf.app.flags.DEFINE_string("vocab_file", None, "Path to vocab file")
tf.app.flags.DEFINE_boolean("pretrained_target", True,
                            "Use pre-trained embeddings for head words.")
tf.app.flags.DEFINE_boolean("pretrained_input", False,
                            "Use pre-trained embeddings for gloss words.")
tf.app.flags.DEFINE_string("embeddings_path",
                           "../embeddings/GoogleWord2Vec.clean.normed.pkl",
                           "Path to pre-trained (.pkl) word embeddings.")
tf.app.flags.DEFINE_string("encoder_type", "recurrent", "BOW or recurrent.")
tf.app.flags.DEFINE_string("model_name", "recurrent", "BOW or recurrent.")



FLAGS = tf.app.flags.FLAGS


def read_data(data_path, vocab_size, phase="train"):
  """Read data from gloss and head files.

  Args:
    data_path: path to the definition .gloss and .head files.
    vocab_size: total number of word types in the data.
    phase: used to locate definitions (train or dev).

  Returns:
    a tuple (gloss, head)
      where gloss is an np array of encoded glosses and head is an
      encoded array of head words; len(gloss) == len(head).
  """
  glosses, heads = [], []
  gloss_path = os.path.join(
      data_path, "%s.definitions.ids%s.gloss" % (phase, vocab_size))
  head_path = os.path.join(
      data_path, "%s.definitions.ids%s.head" % (phase, vocab_size))
  with tf.gfile.GFile(gloss_path, mode="r") as gloss_file:
    with tf.gfile.GFile(head_path, mode="r") as head_file:
      gloss, head = gloss_file.readline(), head_file.readline()
      counter = 0
      while gloss and head:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        gloss_ids = np.array([int(x) for x in gloss.split()], dtype=np.int32)
        glosses.append(gloss_ids)
        heads.append(int(head))
        gloss, head = gloss_file.readline(), head_file.readline()
  return np.asarray(glosses), np.array(heads, dtype=np.int32)


def load_pretrained_embeddings(embeddings_file_path):
  """Loads pre-trained word embeddings.

  Args:
    embeddings_file_path: path to the pickle file with the embeddings.

  Returns:
    tuple of (dictionary of embeddings, length of each embedding).
  """
  print("Loading pretrained embeddings from %s" % embeddings_file_path)
  with open(embeddings_file_path, "rb") as input_file:
    pre_embs_dict = pickle.load(input_file, encoding='bytes')
  iter_keys = iter(pre_embs_dict.keys())
  first_key = next(iter_keys)
  embedding_length = len(pre_embs_dict[first_key])
  print("%d embeddings loaded; each embedding is length %d" %
        (len(pre_embs_dict.values()), embedding_length))
  return pre_embs_dict, embedding_length


def get_embedding_matrix(embedding_dict, vocab, emb_dim):
  emb_matrix = np.zeros([len(vocab), emb_dim])
  for word, ii in vocab.items():
    if word in embedding_dict:
      emb_matrix[ii] = embedding_dict[word]
    else:
      print("OOV word when building embedding matrix: ", word)
  return np.asarray(emb_matrix)


def gen_batch(raw_data, batch_size):
  raw_x, raw_y = raw_data
  data_length = len(raw_x)
  num_batches = data_length // batch_size
  data_x, data_y = [], []
  for i in range(num_batches):
    data_x = raw_x[batch_size * i:batch_size * (i + 1)]
    data_y = raw_y[batch_size * i:batch_size * (i + 1)]
    yield (data_x, data_y)


def gen_epochs(data_path, total_epochs, batch_size, vocab_size, phase="train"):
  # Read all of the glosses and heads into two arrays.
  raw_data = read_data(data_path, vocab_size, phase)
  # Return a generator over the data.
  for _ in range(total_epochs):
    yield gen_batch(raw_data, batch_size)


def build_model(max_seq_len, vocab_size, emb_size, learning_rate, encoder_type,
                pretrained_target=True, pretrained_input=False, pre_embs=None):
  """Build the dictionary model including loss function.

  Args:
    max_seq_len: maximum length of gloss.
    vocab_size: number of words in vocab.
    emb_size: size of the word embeddings.
    learning_rate: learning rate for the optimizer.
    encoder_type: method of encoding (RRN or BOW).
    pretrained_target: Boolean indicating pre-trained head embeddings.
    pretrained_input: Boolean indicating pre-trained gloss word embeddings.
    pre_embs: pre-trained embedding matrix.

  Returns:
    tuple of (gloss_in, head_in, total_loss, train_step, output_form)

  Creates the embedding matrix for the input, which is split into the
  glosses (definitions) and the heads (targets). So checks if there are
  pre-trained embeddings for the glosses or heads, and if not sets up
  some trainable embeddings. The default is to have pre-trained
  embeddings for the heads but not the glosses.

  The encoder for the glosses is either an RNN (with LSTM cell) or a
  bag-of-words model (in which the word vectors are simply
  averaged). For the RNN, the output is the output vector for the
  final state.

  If the heads are pre-trained, the output of the encoder is put thro'
  a non-linear layer, and the loss is the cosine distance. Without
  pre-trained heads, a linear layer on top of the encoder output is
  used to predict logits for the words in the vocabulary, and the loss
  is cross-entropy.
  """
  # Build the TF graph on the GPU.
  with tf.device("/device:GPU:0"):
    tf.reset_default_graph()
    # Batch of input definitions (glosses).
    gloss_in = tf.placeholder(
        tf.int32, [None, max_seq_len], name="input_placeholder")
    # Batch of the corresponding targets (heads).
    head_in = tf.placeholder(tf.int32, [None], name="labels_placeholder")
    with tf.variable_scope("embeddings"):
      if pretrained_input:
        assert pre_embs is not None, "Must include pre-trained embedding matrix"
        # embedding_matrix is pre-trained embeddings.
        embedding_matrix = tf.get_variable(
            name="inp_emb",
            shape=[vocab_size, emb_size],
            initializer=tf.constant_initializer(pre_embs),
            trainable=False)
      else:
        # embedding_matrix is learned.
        embedding_matrix = tf.get_variable(
            name="inp_emb",
            shape=[vocab_size, emb_size])
    # embeddings for the batch of definitions (glosses).
    embs = tf.nn.embedding_lookup(embedding_matrix, gloss_in)
    if pretrained_target:
      out_size = pre_embs.shape[-1]
    else:
      out_size = emb_size
    # RNN encoder for the definitions.
    if encoder_type == "recurrent":
      cell = tf.nn.rnn_cell.LSTMCell(emb_size)
      # state is the final state of the RNN.
      _, state = tf.nn.dynamic_rnn(cell, embs, dtype=tf.float32)
      # state is a pair: (hidden_state, output)
      core_out = state[0]
    else:
      core_out = tf.reduce_mean(embs, axis=1)
    # core_out is the output from the gloss encoder.
    output_form = "cosine"
    if pretrained_target:
      # Create a loss based on cosine distance for pre-trained heads.
      if pretrained_input:
        # Already have the pre-trained embedding matrix, so use that.
        out_emb_matrix = embedding_matrix
      else:
        # Target embeddings are pre-trained.
        out_emb_matrix = tf.get_variable(
            name="out_emb",
            shape=[vocab_size, out_size],
            initializer=tf.constant_initializer(pre_embs),
            trainable=False)
      # Put core_out thro' a final non-linear layer.
      core_out = tf.contrib.layers.fully_connected(
          core_out,
          out_size,
          activation_fn=tf.tanh)
      # Embeddings for the batch of targets/heads.
      targets = tf.nn.embedding_lookup(out_emb_matrix, head_in)
      # cosine_distance assumes the arguments are unit normalized.
      losses = tf.losses.cosine_distance(
          tf.nn.l2_normalize(targets, 1),
          tf.nn.l2_normalize(core_out, 1),
          dim=1)
    else:
      # Create a softmax loss when no pre-trained heads.
      out_emb_matrix = tf.get_variable(
          name="out_emb", shape=[emb_size, vocab_size])
      logits = tf.matmul(core_out, out_emb_matrix)
      pred_dist = tf.nn.softmax(logits, name="predictions")
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=head_in, logits=pred_dist)
      output_form = "softmax"
    # Average loss across batch.
    total_loss = tf.reduce_mean(losses, name="total_loss")
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    return gloss_in, head_in, total_loss, train_step, output_form


def train_network(model, num_epochs, batch_size, data_dir, save_dir,
                  vocab_size, name="model", verbose=True):
  # Running count of the number of training instances.
  num_training = 0
  # saver object for saving the model after each epoch.
  saver = tf.train.Saver()
  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    gloss_in, head_in, total_loss, train_step, _ = model
    # Initialize the model parameters.
    sess.run(tf.global_variables_initializer())
    # Record all training losses for potential reporting.
    training_losses = []
    # epoch is a generator of batches which passes over the data once.
    for idx, epoch in enumerate(
        gen_epochs(
            data_dir, num_epochs, batch_size, vocab_size, phase="train")):
      # Running total for training loss reset every 500 steps.
      training_loss = 0
      if verbose:
        print("\nEPOCH", idx)
      for step, (gloss, head) in enumerate(epoch):
        num_training += len(gloss)
        training_loss_, _ = sess.run(
            [total_loss, train_step],
            feed_dict={gloss_in: gloss, head_in: head})
        training_loss += training_loss_
        if step % 500 == 0 and step > 0:
          if verbose:
            loss_ = training_loss / 500
            print("Average loss step %s, for last 500 steps: %s"
                  % (step, loss_))
          training_losses.append(training_loss / 500)
          training_loss = 0
      # Save current model after another epoch.
      save_path = os.path.join(save_dir, "%s_%s.ckpt" % (name, idx))
      save_path = saver.save(sess, save_path)
      print("Model saved in file: %s after epoch: %s" % (save_path, idx))
    print("Total data points seen during training: %s" % num_training)
    return save_dir, saver

def evaluate_model(sess, data_dir, input_node, target_node, prediction,
                   loss, embs, out_form="cosine"): 
  # read the development and test data using gen_epochs
  # use sess.run and feed_dict to get a prediction
  # (as numpy variable)
  # use numpy to calculate the median rank over 200 dev instances
  # also print out the rank for each of the 200 instances
  # to see where the model does well, and badly!

  print("evaluate_model() needs implementing!!!")


def restore_model(sess, save_dir, vocab_file, out_form):
  model_path = tf.train.latest_checkpoint(save_dir)
  # restore the model from the meta graph
  saver = tf.train.import_meta_graph(model_path + ".meta")
  saver.restore(sess, model_path)
  graph = tf.get_default_graph()
  # get the names of input and output tensors
  input_node = graph.get_tensor_by_name("input_placeholder:0")
  target_node = graph.get_tensor_by_name("labels_placeholder:0")
  if out_form == "softmax":
    predictions = graph.get_tensor_by_name("predictions:0")
  else:
    predictions = graph.get_tensor_by_name("fully_connected/Tanh:0")
  loss = graph.get_tensor_by_name("total_loss:0") # check this is OK
  # vocab is mapping from words to ids, rev_vocab is the reverse.
  vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_file)
  return input_node, target_node, predictions, loss, vocab, rev_vocab

def query_model(sess, input_node, predictions, vocab, rev_vocab, 
                max_seq_len, saver=None, embs=None, out_form="cosine"):
  while True:
    sys.stdout.write("Type a definition: ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    sys.stdout.write("Number of candidates: ")
    sys.stdout.flush()
    top = int(sys.stdin.readline())
    # Get token-ids for the input gloss.
    token_ids = data_utils.sentence_to_token_ids(sentence, vocab)
    # Pad out (or truncate) the input gloss ids.
    padded_ids = np.asarray(data_utils.pad_sequence(token_ids, max_seq_len))
    input_data = np.asarray([padded_ids])
    # Single vector encoding the input gloss.
    model_preds = sess.run(predictions, feed_dict={input_node: input_data})
    # Softmax already provides scores over the vocab.
    if out_form == "softmax":
      # Exclude padding and _UNK tokens from the top-k calculation.
      candidate_ids = np.squeeze(model_preds)[2:].argsort()[-top:][::-1] + 2
      # Replace top-k ids with corresponding words.
      candidates = [rev_vocab[idx] for idx in candidate_ids]
      # Cosine requires sim to be calculated for each vocab word.
    else:
      sims = 1 - np.squeeze(dist.cdist(model_preds, embs, metric="cosine"))
      # replace nans with 0s.
      sims = np.nan_to_num(sims)
      candidate_ids = sims.argsort()[::-1][:top]
      candidates = [rev_vocab[idx] for idx in candidate_ids]
    # get baseline candidates from the raw embedding space.
    base_rep = np.asarray([np.mean(embs[token_ids], axis=0)])
    sims_base = 1 - np.squeeze(dist.cdist(base_rep, embs, metric="cosine"))
    sims_base = np.nan_to_num(sims_base)
    candidate_ids_base = sims_base.argsort()[::-1][:top]
    candidates_base = [rev_vocab[idx] for idx in candidate_ids_base]
    print("Top %s baseline candidates:" % top)
    for ii, cand in enumerate(candidates_base):
      print("%s: %s" % (ii + 1, cand))
    print("\n Top %s candidates from the model:" % top)
    for ii, cand in enumerate(candidates):
      print("%s: %s" % (ii + 1, cand))
    old_model_preds = model_preds
    sys.stdout.flush()
    sentence = sys.stdin.readline()


def main(unused_argv):
  """Calls train and test routines for the dictionary model.

  If restore FLAG is true, loads an existing model and runs test
  routine. If restore FLAG is false, builds a model and trains it.
  """
  if FLAGS.vocab_file is None:
    vocab_file = os.path.join(FLAGS.data_dir,
                              "definitions_%s.vocab" % FLAGS.vocab_size)
  else:
    vocab_file = FLAGS.vocab_file

  # Build and train a dictionary model.
  if not FLAGS.restore:
    emb_size = FLAGS.embedding_size
    # Load any pre-trained word embeddings.
    if FLAGS.pretrained_input or FLAGS.pretrained_target:
      # embs_dict is a dictionary from words to vectors.
      embs_dict, pre_emb_dim = load_pretrained_embeddings(FLAGS.embeddings_path)
      if FLAGS.pretrained_input:
        emb_size = pre_emb_dim
    else:
      pre_embs, embs_dict = None, None

    # Create vocab file, process definitions (if necessary).
    data_utils.prepare_dict_data(
        FLAGS.data_dir,
        FLAGS.train_file,
        FLAGS.dev_file,
        vocabulary_size=FLAGS.vocab_size,
        max_seq_len=FLAGS.max_seq_len)
    # vocab is a dictionary from strings to integers.
    vocab, _ = data_utils.initialize_vocabulary(vocab_file)
    pre_embs = None
    if FLAGS.pretrained_input or FLAGS.pretrained_target:
      # pre_embs is a numpy array with row vectors for words in vocab.
      # for vocab words not in embs_dict, vector is all zeros.
      pre_embs = get_embedding_matrix(embs_dict, vocab, pre_emb_dim)

    # Build the TF graph for the dictionary model.
    model = build_model(
        max_seq_len=FLAGS.max_seq_len,
        vocab_size=FLAGS.vocab_size,
        emb_size=emb_size,
        learning_rate=FLAGS.learning_rate,
        encoder_type=FLAGS.encoder_type,
        pretrained_target=FLAGS.pretrained_target,
        pretrained_input=FLAGS.pretrained_input,
        pre_embs=pre_embs)

    # Run the training for specified number of epochs.
    save_path, saver = train_network(
        model,
        FLAGS.num_epochs,
        FLAGS.batch_size,
        FLAGS.data_dir,
        FLAGS.save_dir,
        FLAGS.vocab_size,
        name=FLAGS.model_name)

  # Load an existing model.
  else:
    # Note cosine loss output form is hard coded here. For softmax output
    # change "cosine" to "softmax"
    if FLAGS.pretrained_input or FLAGS.pretrained_target:
      embs_dict, pre_emb_dim = load_pretrained_embeddings(FLAGS.embeddings_path)
      vocab, _ = data_utils.initialize_vocabulary(vocab_file)
      pre_embs = get_embedding_matrix(embs_dict, vocab, pre_emb_dim)

    with tf.device("/cpu:0"):
      with tf.Session() as sess:
        (input_node, target_node, predictions, loss, vocab,
          rev_vocab) = restore_model(sess, FLAGS.save_dir, vocab_file,
                                     out_form="cosine")

        if FLAGS.evaluate:
          evaluate_model(sess, FLAGS.data_dir,
                         input_node, target_node,
                         predictions, loss, embs=pre_embs, out_form="cosine")

        # Load the final saved model and run querying routine.
        query_model(sess, input_node, predictions,
                    vocab, rev_vocab, FLAGS.max_seq_len, embs=pre_embs,
                    out_form="cosine")


if __name__ == "__main__":
  tf.app.run()
