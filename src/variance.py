import numpy as np
import itertools
from joblib import Parallel, delayed
import iisignature
from tqdm.auto import tqdm

from shuffle import shuffle


CACHE = {}

def _sig(p, order):
    return np.r_[1., iisignature.sig(p, order)]


def get_basis(dim, order):
    alphabet = range(dim)

    basis = [itertools.product(*([alphabet] * n)) for n in range(order + 1)]
    basis = list(itertools.chain(*basis))

    return basis


def _build_row(w, basis, E):
    Ai = []
    for j, v in enumerate(basis):
        z = shuffle(w, v)
        Ai.append(sum(E[z_] for z_ in z))

    return np.array(Ai)

def _build_matrix(basis, E):
    shuffled = {}

    A = np.zeros((len(basis), len(basis)))

    pbar = tqdm(basis, total=len(basis), desc="Building shuffle matrix")
    A = np.array(Parallel(n_jobs=-1)(delayed(_build_row)(w, basis, E) for w in pbar))

    A_inv = np.linalg.pinv(A)

    return A_inv

def prepare(corpus, order):
    dim = corpus[0].shape[1]
    basis = get_basis(dim, order)
    basis_extended = get_basis(2, 2 * order)

    sigs = np.array(Parallel(n_jobs=-1)(delayed(_sig)(p, 2 * order) for p in tqdm(corpus, desc="Computing signatures")))
    E = dict(zip(basis_extended, np.mean(sigs, axis=0)))

    A_inv = _build_matrix(basis, E)

    CACHE[hash(str((order, corpus)))] = (sigs, A_inv)


def variance(paths, corpus, order):
    if hash(str((order, corpus))) not in CACHE:
        print("Preparing...")
        prepare(corpus, order)
        print("Done.")

    sigs, A_inv = CACHE[hash(str((order, corpus)))]

    res = []

    for path in tqdm(paths, desc="Computing variances"):
        #a = _sig(path, order).reshape(-1, 1)
        #res.append(np.dot(a.T, A_inv).dot(a)[0, 0])

        sig = _sig(path, order)
        a = sig - sigs[:, :len(sig)]
        res.append(np.diag(np.dot(a, A_inv).dot(a.T)).min())

    return res
