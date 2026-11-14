#!/usr/bin/env python3
"""The neural-network filter of the staged screen (tranche strategy).

AdaptiveFlow trains a small network on the prescreen's docking scores and uses it to drop molecules of the selected
tranches that are unlikely to dock well, so the primary screen docks fewer of them. This module is that classifier,
written with numpy (the tool's environment has no PyTorch; adding it would grow the environment by 1.3 GB), following
the recipe of AFVS's templates/ml_classifier.py (Morgan fingerprints of 1024 bits and radius 2; a network of
1024-512-256-128-1 nodes with ReLU and a sigmoid output; the best 25% of the training scores are the positives; the
majority class is undersampled; Adam with a rate of 0.001, batches of 256, at most 50 epochs, early stopping after 5
epochs without improvement on 10% of the data). Models are not interchangeable with the PyTorch ones.

Added to the recipe: part of the data is held back to measure the model (area under the ROC curve, and the share of the
good molecules that the cutoff keeps), because a model that is no better than chance must not be allowed to drop
molecules; and the training set is capped, so that a prescreen of millions of forms trains in minutes.
"""
import os

import numpy as np

FP_BITS = 1024
FP_RADIUS = 2
LAYERS = (FP_BITS, 512, 256, 128, 1)
TOP_PERCENT = 25                # the best 25% of the training scores are the positive class
BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 1e-3
VALIDATION_FRACTION = 0.10      # of the (undersampled) training data, for early stopping
HOLDOUT_FRACTION = 0.15         # of all the data, never trained on: measures the model
MAX_TRAIN = 200_000             # forms used at most (a random sample beyond that)
PREDICT_BATCH = 4096


def _generator():
    from rdkit.Chem import rdFingerprintGenerator
    return rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def fingerprints(smiles, generator=None):
    """Morgan fingerprints of SMILES as a float32 matrix, and a mask of the rows that could be made (empty or unreadable
    SMILES give a zero row and False)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    generator = generator or _generator()
    matrix = np.zeros((len(smiles), FP_BITS), dtype=np.float32)
    ok = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            matrix[i] = generator.GetFingerprintAsNumPy(mol)
            ok[i] = True
    return matrix, ok


def auc(labels, scores):
    """Area under the ROC curve (the chance that a positive scores above a negative; ties count half). None when one
    class is missing."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores))
    start = 0
    while start < len(scores):                      # average ranks for ties
        end = start
        while end + 1 < len(scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


class Network:
    """The fully connected network, trained by Adam on the binary cross-entropy."""

    def __init__(self, rng=None, weights=None, biases=None):
        if weights is None:
            rng = rng or np.random.default_rng(0)
            weights = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(LAYERS[:-1], LAYERS[1:])]
            biases = [np.zeros(b, np.float32) for b in LAYERS[1:]]
        self.weights, self.biases = weights, biases
        self._m = [np.zeros_like(p) for p in self.weights + self.biases]
        self._v = [np.zeros_like(p) for p in self.weights + self.biases]
        self._t = 0

    def forward(self, x):
        activations = [x]
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            activations.append(1 / (1 + np.exp(-np.clip(z, -30, 30))) if i == last else np.maximum(z, 0))
        return activations

    def predict(self, x):
        return self.forward(x)[-1][:, 0]

    def step(self, x, y, rate=LEARNING_RATE):
        activations = self.forward(x)
        grad = ((activations[-1][:, 0] - y) / len(y))[:, None]
        grad_w, grad_b = [], []
        for i in reversed(range(len(self.weights))):
            grad_w.insert(0, activations[i].T @ grad)
            grad_b.insert(0, grad.sum(0))
            if i:
                grad = (grad @ self.weights[i].T) * (activations[i] > 0)
        self._t += 1
        parameters = self.weights + self.biases
        for k, g in enumerate(grad_w + grad_b):
            self._m[k] = 0.9 * self._m[k] + 0.1 * g
            self._v[k] = 0.999 * self._v[k] + 0.001 * g * g
            m_hat = self._m[k] / (1 - 0.9 ** self._t)
            v_hat = self._v[k] / (1 - 0.999 ** self._t)
            parameters[k] -= rate * m_hat / (np.sqrt(v_hat) + 1e-8)


def _loss(p, y):
    return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))


class Model:
    """A trained network with what is needed to use it: the cutoff score of the positives and the training report."""

    def __init__(self, network, threshold, report):
        self.network, self.threshold, self.report = network, threshold, report

    def predict(self, smiles, generator=None):
        """Probability of being a good binder for every SMILES (NaN where the SMILES cannot be read)."""
        out = np.full(len(smiles), np.nan, dtype=np.float32)
        generator = generator or _generator()
        for start in range(0, len(smiles), PREDICT_BATCH):
            chunk = smiles[start:start + PREDICT_BATCH]
            x, ok = fingerprints(chunk, generator)
            if ok.any():
                out[start:start + len(chunk)][ok] = self.network.predict(x[ok])
        return out

    def save(self, path):
        arrays = {f"w{i}": w for i, w in enumerate(self.network.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.network.biases)})
        arrays["threshold"] = np.array(self.threshold)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            n = len(LAYERS) - 1
            network = Network(weights=[data[f"w{i}"] for i in range(n)], biases=[data[f"b{i}"] for i in range(n)])
            return cls(network, float(data["threshold"]), {})


def train(smiles, scores, cutoff=0.5, seed=1, max_train=MAX_TRAIN, top_percent=TOP_PERCENT):
    """Train a classifier on (SMILES, docking score) pairs. Lower scores are better. Returns a Model whose `report`
    says how many forms were used and how good the model is on forms it never saw: `auc`, `recall` (share of the
    positives with a probability of at least `cutoff`) and `kept` (share of all held-back forms that reach the cutoff;
    a model that keeps 49% and recalls 94% is useful, one that keeps 50% and recalls 50% is not)."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=np.float64)
    if len(smiles) > max_train:
        pick = rng.choice(len(smiles), max_train, replace=False)
        smiles, scores = [smiles[i] for i in pick], scores[pick]
    x, ok = fingerprints(list(smiles))
    x, scores = x[ok], scores[ok]
    if len(scores) < 20:
        raise ValueError("fewer than 20 readable molecules to train on")
    threshold = float(np.percentile(scores, top_percent))
    y = (scores <= threshold).astype(np.float32)

    def split(indices, fraction):
        """Stratified: `fraction` of each class, at least one of each when the class has two or more."""
        taken = []
        for label in (0, 1):
            members = indices[y[indices] == label]
            members = members[rng.permutation(len(members))]
            taken.append(members[:max(1, int(round(fraction * len(members)))) if len(members) > 1 else 0])
        chosen = np.concatenate(taken)
        return chosen, np.setdiff1d(indices, chosen)

    everyone = np.arange(len(y))
    holdout, rest = split(everyone, HOLDOUT_FRACTION)
    positives, negatives = rest[y[rest] == 1], rest[y[rest] == 0]
    if len(positives) < 2 or len(negatives) < 2:
        raise ValueError("too few molecules of one class to train on")
    count = min(len(positives), len(negatives))                      # undersample the majority class
    balanced = np.concatenate([rng.choice(positives, count, replace=False), rng.choice(negatives, count, replace=False)])
    validation, training = split(balanced, VALIDATION_FRACTION)
    network = Network(rng)
    best_loss, best, waited, epochs, best_epoch = np.inf, None, 0, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        epochs = epoch
        order = training[rng.permutation(len(training))]
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            network.step(x[batch], y[batch])
        loss = _loss(network.predict(x[validation]), y[validation])
        if loss < best_loss - 1e-6:
            best_loss, best_epoch, waited = loss, epoch, 0
            best = ([w.copy() for w in network.weights], [b.copy() for b in network.biases])
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    network.weights, network.biases = best
    p = network.predict(x[holdout])
    held = y[holdout].astype(bool)
    report = {"forms": int(len(scores)), "positives": int(y.sum()), "score_threshold": threshold,
              "trained_on": int(len(training)), "held_back": int(len(holdout)), "epochs": epochs,
              "best_epoch": best_epoch, "validation_loss": float(best_loss), "auc": auc(held, p),
              "cutoff": cutoff, "kept": float((p >= cutoff).mean()) if len(p) else None,
              "recall": float((p[held] >= cutoff).mean()) if held.any() else None}
    return Model(network, threshold, report)
