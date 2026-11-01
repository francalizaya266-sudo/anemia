"""Clustered ROC statistics for the second internal review.
obuchowski(y, P, cl): AUCs, covariance matrix of the AUCs for clustered data (Obuchowski 1997, Biometrics 53:567-578),
  P is (k, n) predictions of k models on the same n rows, cl the cluster id of each row (duplicate group).
cluster_boot(y, P, cl): cluster (group) bootstrap of each AUC and of the pairwise difference P[0]-P[1].
"""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

def _psi_components(y, p):
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    xp = p[pos][:, None]; yn = p[neg][None, :]
    psi = (xp > yn) + 0.5 * (xp == yn)            # M x N
    return pos, neg, psi.mean(1), psi.mean(0)       # V10 for each positive, V01 for each negative

def obuchowski(y, P, cl):
    y = np.asarray(y); P = np.atleast_2d(np.asarray(P, float)); cl = np.asarray(cl)
    ids, inv = np.unique(cl, return_inverse=True); I = len(ids); k = P.shape[0]
    M = int((y == 1).sum()); N = int((y == 0).sum())
    m_i = np.bincount(inv, weights=(y == 1), minlength=I); n_i = np.bincount(inv, weights=(y == 0), minlength=I)
    th = np.zeros(k); V10 = np.zeros((k, I)); V01 = np.zeros((k, I))
    for r in range(k):
        pos, neg, v10, v01 = _psi_components(y, P[r]); th[r] = v10.mean()
        V10[r] = np.bincount(inv[pos], weights=v10, minlength=I); V01[r] = np.bincount(inv[neg], weights=v01, minlength=I)
    A = V10 - m_i[None] * th[:, None]; B = V01 - n_i[None] * th[:, None]
    f = I / (I - 1)
    S10 = f * (A @ A.T) / M; S01 = f * (B @ B.T) / N; S11 = f * (A @ B.T)
    C = S10 / M + S01 / N + (S11 + S11.T) / (M * N)
    return th, C

def obu_ci(y, p, cl, alpha=0.05):
    th, C = obuchowski(y, p, cl); se = np.sqrt(C[0, 0]); z = stats.norm.ppf(1 - alpha / 2)
    return float(th[0]), [float(th[0] - z * se), float(th[0] + z * se)], float(se)

def obu_test(y, p1, p2, cl):
    th, C = obuchowski(y, np.vstack([p1, p2]), cl); d = th[0] - th[1]; se = np.sqrt(C[0, 0] + C[1, 1] - 2 * C[0, 1])
    z = d / se; return float(d), [float(d - 1.96 * se), float(d + 1.96 * se)], float(z), float(2 * stats.norm.sf(abs(z)))

def cluster_boot(y, P, cl, B=2000, seed=42):
    y = np.asarray(y); P = np.atleast_2d(np.asarray(P, float)); cl = np.asarray(cl)
    ids, inv = np.unique(cl, return_inverse=True); I = len(ids)
    rows = [np.where(inv == g)[0] for g in range(I)]; rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        idx = np.concatenate([rows[g] for g in rng.integers(0, I, I)])
        if len(np.unique(y[idx])) < 2: continue
        out.append([roc_auc_score(y[idx], P[r, idx]) for r in range(P.shape[0])])
    out = np.array(out); res = {'auc_ci': [[float(np.percentile(out[:, r], 2.5)), float(np.percentile(out[:, r], 97.5))] for r in range(P.shape[0])]}
    if P.shape[0] >= 2:
        dd = out[:, 0] - out[:, 1]
        res['diff_ci'] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]; res['diff_p'] = float(min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean())))
    return res
