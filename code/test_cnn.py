import tensorflow as tf, numpy as np

def run_cnn(context_char_embs, context_embs, qn_char_embs, qn_embs):
    word_len = 3
    kernel_size = (1, 2)  # (num_words, window_size)
    num_output_states = 3
    pool_size = (1, word_len)

    # Character-level CNN for context
    context_cnn_output = tf.layers.conv2d(context_char_embs, num_output_states, kernel_size,
                                          padding='same',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='conv',
                                          reuse=False)  # (batch_size, context_len, word_len, char_embedding_size)

    context_pool = tf.layers.max_pooling2d(context_cnn_output, pool_size, strides=1,
                                           padding='valid')  # (batch_size, context_len, 1, num_output_states)

    context_char_embs = tf.squeeze(context_pool)  # (batch_size, context_len, num_output_states)
    context_embs = tf.concat([context_embs, context_char_embs],
                             axis=2)  # (batch_size, context_len, embedding_size + num_output_states)

    # Character-level CNN for question
    question_cnn_output = tf.layers.conv2d(qn_char_embs, num_output_states, kernel_size, padding='same',
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           name='conv',
                                           reuse=True)  # (batch_size, question_len, word_len, num_output_states)

    question_pool = tf.layers.max_pooling2d(question_cnn_output, pool_size, strides=1,
                                            padding='valid')  # (batch_size, question_len, 1, num_output_states)

    question_char_embs = tf.squeeze(question_pool)  # (batch_size, question_len, num_output_states)
    qn_embs = tf.concat([qn_embs, question_char_embs],
                        axis=2)  # (batch_size, context_len, embedding_size + num_output_states)

    return context_embs, qn_embs


def test_cnn():
    a, b, c, pad = [0.0, 0.1, 0.2], [1, 0.9, 0.8], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]
    word1 = [a, b, pad]
    word2 = [a, b, c]
    word3 = [b, pad, pad]
    padword = [pad, pad, pad]
    char_context1 = [word1, word2, word1, word3]
    char_context2 = [word3, word2, padword, padword]
    char_batch = [char_context1, char_context2]
    context_char_embs = np.array(char_batch)
    context_char_embs = tf.Variable(context_char_embs, dtype=tf.float32)

    word_emb1 = [0.1, 0.1, 0.1, 0.1, 0.1]
    word_emb2 = [0.2, 0.2, 0.2, 0.2, 0.2]
    word_emb3 = [0.3, 0.3, 0.3, 0.3, 0.3]
    padword_emb = [0.0, 0.0, 0.0, 0.0, 0.0]
    glove_context1 = [word_emb1, word_emb2, word_emb1, word_emb3]
    glove_context2 = [word_emb3, word_emb2, padword_emb, padword_emb]
    context_embs = np.array([glove_context1, glove_context2])
    context_embs = tf.Variable(context_embs, dtype=tf.float32)

    context_output, qn_output = run_cnn(context_char_embs, context_embs, context_char_embs, context_embs)

    output_shape = context_output.get_shape()  # (batch_size, context_size, glove_length + num_outputs)
    assert output_shape == (len(char_batch), len(char_context1), 3 + len(word_emb1))
    print 'All tests passed!'

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(context_output)
        print context_output

if __name__ == '__main__':
    test_cnn()
