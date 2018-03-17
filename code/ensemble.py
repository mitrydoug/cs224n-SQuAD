
from argparse import Namespace
import numpy as np

class ModelEnsemble(object):
    def __init__(self, qa_models, FLAGS, method='max_range_prob_sum'):
        self.qa_models = qa_models
        self.FLAGS = FLAGS
        self.method = method

    def get_start_end_pos(self, session, batch):
        if self.method == 'max_range_prob_sum':
            range_dists = ([qa_model.get_start_end_pos(session, batch, return_range_probs=True)
                            for qa_model in self.qa_models])

            range_prob_sums = (
                np.array([
                    np.array([range_dist[batch_num] for range_dist in range_dists])
                      .sum(axis=0).flatten()
                for batch_num in range(batch.batch_size)])

            best_ranges = np.argmax(range_prob_sums, axis=1)
            start_pos = best_ranges // self.FLAGS.context_len
            end_pos = best_ranges % self.FLAGS.context_len
        elif self.method == 'max_range_prob':
            range_dists = ([qa_model.get_start_end_pos(session, batch, return_range_probs=True)
                            for qa_model in self.qa_models])

            range_prob_sums = (
                np.array([
                    np.array([range_dist[batch_num] for range_dist in range_dists])
                      .max(axis=0).flatten()
                for batch_num in range(batch.batch_size)])

            best_ranges = np.argmax(range_prob_sums, axis=1)
            start_pos = best_ranges // self.FLAGS.context_len
            end_pos = best_ranges % self.FLAGS.context_len
        return start_pos, end_pos

def test_ensemble():

    class FakeModel(object):
        def __init__(self, s, e):
            self.s = s
            self.e = e
        def get_start_end_pos(self, session, batch):
            return self.s, self.e

    s1 = np.array([[0.50, 0.5],
                   [0.30, 0.7],
                   [0.25, 0.75]])

    s2 = np.array([[0.1, 0.9],
                   [0.2, 0.8],
                   [0.3, 0.7]])

    s3 = np.array([[0.4, 0.6],
                   [0.2, 0.8],
                   [0.1, 0.9]])

    e1 = np.array([[0.7, 0.3],
                   [0.1, 0.9],
                   [0.9, 0.1]])

    e2 = np.array([[0.4, 0.6],
                   [0.5, 0.5],
                   [0.5, 0.5]])

    e3 = np.array([[0.1, 0.9],
                   [0.2, 0.8],
                   [0.3, 0.7]])

    fake1 = FakeModel(s1, e1)
    fake2 = FakeModel(s2, e2)
    fake3 = FakeModel(s3, e3)

    FLAGS = Namespace(context_len=2)
    batch = Namespace(batch_size=3)
    ensemble = ModelEnsemble([fake1, fake2, fake3], FLAGS)
    start, end = ensemble.get_start_end_pos(None, batch)
    print start, end

if __name__ == '__main__':
    test_ensemble()
