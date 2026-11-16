import pickle, sys, time, json
from multiprocessing import Pool
sys.path.insert(0, '.')
import numpy as np

def job(args):
    import engine as en
    code, seed, pol, params = args
    I = pickle.load(open('../data/processed/instances_v2.pkl', 'rb'))
    inst = I['case'] if code == 'CASE' else I['pool'][code]
    sc = en.make_scenario(inst, seed)
    out = {}
    for fast in (False, True):
        en.USE_FAST = fast
        t0 = time.time()
        r = en.simulate(inst, sc, en.Policy(pol, params))
        out[fast] = ({k: v for k, v in r.items() if isinstance(v, float)}, time.time() - t0)
    diff = [k for k in out[False][0] if not (out[False][0][k] == out[True][0][k] or (np.isnan(out[False][0][k]) and np.isnan(out[True][0][k])))]
    return code, seed, pol, diff, round(out[False][1], 2), round(out[True][1], 2)

if __name__ == '__main__':
    tasks = []
    for code, seed in (('C2013-09', 20001), ('C2016-12', 20002), ('CASE', 900005), ('C2014-06', 600003), ('C2017-01', 600004), ('C2015-29', 20003)):
        for pol, params in (('rssp', {'delay_premium': 0.5}), ('rssp_memoryless', {'delay_premium': 0.5}), ('nonstrategic_mpc', {}), ('centralized', {})):
            tasks.append((code, seed, pol, params))
    with Pool(3) as p:
        res = p.map(job, tasks)
    bad = 0
    for r in res:
        print(r); bad += len(r[3]) > 0
    print('MISMATCHES', bad)
