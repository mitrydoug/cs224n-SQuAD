import json
import os

import numpy as np

if __name__ == '__main__':

    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    POWERS=np.arange(0.0, 1.1, 0.1)
    F1={}
    EM={}
    best_f1 = 0.0
    best_em = 0.0
    for p in POWERS:
        os.system(('python code/main.py --ans_len_dist_power={} '
                   '--mode=official_eval '
                   '--json_in_path=data/dev-v1.1.json '
                   '--ckpt_load_dir=experiments/baseline/best_checkpoint')
                   .format(p))
        os.system('python code/evaluate.py data/dev-v1.1.json '
                  'predictions.json > performance.json')
        with open('performance.json', 'r') as f:
            scores = json.load(f)
        F1[p] = scores['f1']
        EM[p] = scores['exact_match']
        if F1[p] > F1[best_f1]:
            best_f1 = p
        if EM[p] > EM[best_em]:
            best_em = p
    with open('f1-1.json', 'w') as f:
        json.dump(F1, f)
    with open('em-1.json', 'w') as f:
        json.dump(EM, f)
    POWERS=np.arange(best_f1-0.09, best_f1+0.1, 0.01)
    for p in POWERS:
        if p in F1: continue
        os.system(('python code/main.py --ans_len_dist_power={} '
                   '--mode=official_eval '
                   '--json_in_path=data/dev-v1.1.json '
                   '--ckpt_load_dir=experiments/baseline/best_checkpoint')
                   .format(p))
        os.system('python code/evaluate.py data/dev-v1.1.json '
                  'predictions.json > performance.json')
        with open('performance.json', 'r') as f:
            scores = json.load(f)
        F1[p] = scores['f1']
        EM[p] = scores['exact_match']
        if F1[p] > F1[best_f1]:
            best_f1 = p
        if EM[p] > EM[best_em]:
            best_em = p
    with open('f1-2.json', 'w') as f:
        json.dump(F1, f)
    with open('em-2.json', 'w') as f:
        json.dump(EM, f)
    POWERS=np.arange(best_em-0.09, best_em+0.1, 0.01)
    for p in POWERS:
        if p in EM: continue
        os.system(('python code/main.py --ans_len_dist_power={} '
                   '--mode=official_eval '
                   '--json_in_path=data/dev-v1.1.json '
                   '--ckpt_load_dir=experiments/baseline/best_checkpoint')
                   .format(p))
        os.system('python code/evaluate.py data/dev-v1.1.json '
                  'predictions.json > performance.json')
        with open('performance.json', 'r') as f:
            scores = json.load(f)
        F1[p] = scores['f1']
        EM[p] = scores['exact_match']
        if F1[p] > F1[best_f1]:
            best_f1 = p
        if EM[p] > EM[best_em]:
            best_em = p
    with open('f1-3.json', 'w') as f:
        json.dump(F1, f)
    with open('em-3.json', 'w') as f:
        json.dump(EM, f)
