import tensorflow as tf

from tensorflow.contrib import framework, layers, learn, lookup, metrics


def sparse_tensor_dim_lengths(st, axis=0):
  return tf.reduce_sum(
      tf.sparse_to_dense(
          st.indices,
          st.dense_shape,
          tf.ones_like(st.values, dtype=tf.int32)
      ),
      axis=axis
  )


def training_input_fn(file_pattern,
                      batch_size,
                      num_classes,
                      **kwargs):
  def _input_fn():
    features = learn.read_batch_record_features(
        file_pattern,
        batch_size,
        {
            'sentences': tf.VarLenFeature(tf.int64),
            'subreddit': tf.FixedLenFeature([1], tf.int64)
        },
        **kwargs
    )
    labels = {'subreddit': features.pop('subreddit')}
    features['sentence_lengths'] = sparse_tensor_dim_lengths(
        features['sentences'], axis=1)
    features['sentences'] = tf.sparse_tensor_to_dense(features['sentences'])
    return learn.InputFnOps(features, labels, None)
  return _input_fn


def serving_input_fn(vocab_filename,
                     vocab_size=None,
                     default_batch_size=None):
  def _input_fn():
    serialized_tf_example = tf.placeholder(
        dtype=tf.string,
        shape=[default_batch_size],
        name='input_example_tensor'
    )
    features = tf.parse_example(
        serialized_tf_example,
        {'sentences': tf.VarLenFeature(tf.string)}
    )
    index = lookup.HashTable(lookup.TextFileInitializer(
        vocab_filename,
        tf.string, 0,  # Keys are strings in the first column
        tf.int64, lookup.TextFileIndex.LINE_NUMBER,  # values are lines
        delimiter=',',
        vocab_size=vocab_size,
    ), -1)
    # Add one so unknown keys are 0
    features['sentences'].values = index.lookup(
        features['sentences'].values
    ) + 1
    features['sentence_lengths'] = sparse_tensor_dim_lengths(
        features['sentences'], axis=1)
    features['sentences'] = tf.sparse_tensor_to_dense(features['sentences'])
    return learn.InputFnOps(
        features, None, {'examples': serialized_tf_example})
  return _input_fn


def make_model_fn(num_partitions,
                  vocab_size,
                  learning_rate,
                  embedding_size,
                  num_classes):
  def _model_fn(features, labels, mode):
    # Shape [batch_size, max_sent_len]
    # Where max_sent_len is the longest sentence in the batch
    sentences_padded = features['sentences']
    subreddits = labels['subreddit']
    with tf.device(tf.train.replica_device_setter()):
      with tf.variable_scope('embeddings',
                             partitioner=tf.fixed_size_partitioner(
                                 num_partitions)):
        embeddings = tf.get_variable(
            'embeddings',
            shape=[vocab_size + 1, embedding_size]
        )
        lstm = tf.nn.rnn_cell.LSTMCell(num_classes)

    # Shape [batch_size, max_sent_len, embedding_size]
    word_embeddings = tf.nn.embedding_lookup(embeddings, sentences_padded)

    # Shape [batch_size, max_sent_len, num_classes],
    #       [batch_size, num_classes]
    _, final_state = tf.nn.dynamic_rnn(
        lstm,
        word_embeddings,
        dtype=tf.float32,
        scope='rnn',
        sequence_length=features['sentence_length']
    )

    if mode in [learn.ModeKeys.EVAL, learn.ModeKeys.TRAIN]:
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=subreddits,
          logits=final_state
      )

    if mode in [learn.ModeKeys.EVAL, learn.ModeKeys.INFER]:
      predictions = tf.nn.softmax(final_state)

    if mode == learn.ModeKeys.EVAL:
      accuracy = metrics.streaming_accuracy(
          tf.argmax(predictions, axis=1),
          subreddits
      )
      precision = metrics.streaming_precision(predictions, subreddits)
      mean_loss = metrics.streaming_mean(loss)

    if mode == learn.ModeKeys.TRAIN:
      train_op = layers.optimize_loss(
          loss,
          framework.get_or_create_global_step(),
          learning_rate,
          tf.train.AdamOptimizer,
          clip_gradients=1.0,
      )
    return learn.ModelFnOps(
        mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=[accuracy, precision, mean_loss]
    )
  return _model_fn
