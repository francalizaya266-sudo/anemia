# Third round (threshold sensitivity of the stage-4 map): photometric verification of the candidate pairs with 8 or 9
# RANSAC inliers in the SIFT search (the main run verified only pairs with >= 10), same procedure as review2_geomverify.py.
import sys, time; sys.path.insert(0, '/home/claude/exp')
import numpy as np
from multiprocessing import Pool
import review2_geomverify as GV
OUT = '/home/claude/exp/out/review2/'; OUT3 = '/home/claude/exp/out/review3/'
if __name__ == '__main__':
    t0 = time.time(); arr = np.load(OUT + 'orb_pairs.npy'); G = np.load(OUT + 'geomverify.npy')
    done = set(map(tuple, G[:, :2].astype(int).tolist()))
    lo = arr[(arr[:, 3] >= 8) & (arr[:, 3] < 10)][:, :2].astype(int)
    todo = [tuple(p) for p in lo.tolist() if tuple(p) not in done]; print('pairs to verify', len(todo), 'of', len(lo), flush=True)
    with Pool(4, initializer=GV.init, initargs=(GV.reps.image_path.tolist(),)) as pool: res = pool.map(GV.work, todo, chunksize=50)
    np.save(OUT3 + 'geomverify_lo.npy', np.array(res, dtype=float)); print('done %.0fs' % (time.time() - t0), flush=True)
