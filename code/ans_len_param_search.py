import json
import os
import argparse

import numpy as np

if __name__ == '__main__':

    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--step')
    parser.add_argument('--down')
    args = parser.parse_args()
    print args


    exp = args.experiment
    start = float(args.start)
    end = float(args.end)
    step = float(args.step)
    down = float(args.down)

    POWERS=np.arange(start, end+step/2., step)
    F1 = {}
    EM = {}
    best_f1 = 0.0
    best_em = 0.0
    #for p in POWERS:
    #    print 'Testing power={}'.format(p)
    #    if p not in F1:
    #        os.system(('python code/main.py --ans_len_dist_power={} '
    #                   '--mode=official_eval '
    #                   '--json_in_path=data/dev-v1.1.json '
    #                   '--ckpt_load_dir=experiments/{}/best_checkpoint')
    #                   .format(p, exp))
    #        os.system('python code/evaluate.py data/dev-v1.1.json '
    #                  'predictions.json > performance.json')
    #        with open('performance.json', 'r') as f:
    #            scores = json.load(f)
    #    F1[p] = scores['f1']
    #    EM[p] = scores['exact_match']
    #    if F1[p] > F1[best_f1]:
    #        best_f1 = p
    #    if EM[p] > EM[best_em]:
    #        best_em = p
    #    print 'Best F1: {} @ {}'.format(F1[best_f1], best_f1)
    #    print 'Best EM: {} @ {}'.format(EM[best_em], best_em)
    #with open('f1-1.json', 'w') as f:
    #    json.dump(F1, f)
    #with open('em-1.json', 'w') as f:
    #    json.dump(EM, f)


    POWERS=np.arange(best_f1-step+step/down, best_f1+step, step/down)
    for p in POWERS:
        print 'Testing power={}'.format(p)
        if p in F1: continue
        os.system(('python code/main.py --ans_len_dist_power={} '
                   '--mode=official_eval '
                   '--json_in_path=data/dev-v1.1.json '
                   '--ckpt_load_dir=experiments/{}/best_checkpoint')
                   .format(p, exp))
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
        print 'Best F1: {} @ {}'.format(F1[best_f1], best_f1)
        print 'Best EM: {} @ {}'.format(EM[best_em], best_em)
    with open('f1-2.json', 'w') as f:
        json.dump(F1, f)
    with open('em-2.json', 'w') as f:
        json.dump(EM, f)

    POWERS=np.arange(best_em-step+step/down, best_em+step, step/down)
    for p in POWERS:
        print 'Testing power={}'.format(p)
        if p in EM: continue
        os.system(('python code/main.py --ans_len_dist_power={} '
                   '--mode=official_eval '
                   '--json_in_path=data/dev-v1.1.json '
                   '--ckpt_load_dir=experiments/{}/best_checkpoint')
                   .format(p, exp))
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
        print 'Best F1: {} @ {}'.format(F1[best_f1], best_f1)
        print 'Best EM: {} @ {}'.format(EM[best_em], best_em)
    with open('f1-3.json', 'w') as f:
        json.dump(F1, f)
    with open('em-3.json', 'w') as f:
        json.dump(EM, f)
