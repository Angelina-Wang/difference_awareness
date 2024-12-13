import pickle
import numpy as np

cats = ['D', 'N']
to_print = False

def d_check(diff_answers, equal_answers):
    assert 2 not in diff_answers
    assert 1 in diff_answers
    assert 0 in diff_answers
    assert np.mean(equal_answers) == 2
def np_check(diff_answers, equal_answers):
    assert 2 not in diff_answers
    assert 1 in diff_answers
    assert 0 in diff_answers
    assert 2 not in equal_answers
    assert 1 in equal_answers
    assert 0 in equal_answers

for cat in cats:
    print("--{}--".format(cat))
    for i in range(1, 5):
        try:
            diffs, equals = pickle.load(open('{0}{1}_1k.pkl'.format(cat, i), 'rb'))
            print("{0} 1k: {1} ({2} scen) in D, {3} ({4} scen) in E".format(i, len(diffs), len(np.unique([part[2] for part in diffs])), len(equals), len(np.unique([part[2] for part in equals]))))
            assert len(np.unique([part[0] for part in diffs])) == len(diffs), np.unique([part[0] for part in diffs], return_counts=True)
            assert len(np.unique([part[0] for part in equals])) == len(equals), np.unique(np.unique([part[0] for part in equals], return_counts=True)[1], return_counts=True)
            diff_answers = [part[1] for part in diffs]
            equal_answers = [part[1] for part in equals]
            if to_print:
                inds = np.random.choice(np.arange(len(diffs)), size=3, replace=False)
                for ind in inds:
                    print(diffs[ind])
                inds = np.random.choice(np.arange(len(equals)), size=3, replace=False)
                print()
                for ind in inds:
                    print(equals[ind])
            if 'D' in cat:
                d_check(diff_answers, equal_answers)
            else:
                np_check(diff_answers, equal_answers)
        except FileNotFoundError:
            print('{0}{1}_1k.pkl does not exist'.format(cat, i))

