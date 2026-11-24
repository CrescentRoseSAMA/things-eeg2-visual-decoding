"""Use visual class neighbours to choose extra frozen images for forward InfoNCE."""
import numpy as np


def neighbors(centers):
    unit = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    return np.argsort(-similarity, axis=1, kind="stable")[:, :-1]


class ImageNegatives:
    def __init__(self, images, labels, mode):
        if mode not in ("random", "near"):
            raise ValueError("Expected random or near")
        self.mode = mode
        classes, self.labels, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if len(classes) <= 20 or not np.all(counts == 10):
            raise ValueError("Expected at least 21 classes with ten images each")
        self.rows = np.argsort(self.labels, kind="stable").reshape(-1, 10)
        means = images[self.rows].mean(1)
        self.ordered = neighbors(means)

    def sample(self, order, batch_size, seed, epoch):
        rng = np.random.default_rng([seed, epoch, 621])
        pool = self.ordered[:, :20] if self.mode == "near" else self.ordered
        preferred = rng.permuted(pool, axis=1)[:, :16]
        batch_id = np.empty(len(order), dtype=np.int64)
        batch_id[order] = np.arange(len(order)) // batch_size
        slots = batch_id[self.rows]
        full = np.where((slots == slots[:, :1]).all(1), slots[:, 0], -1)
        result = np.empty((len(order), 16), dtype=np.int64)
        for start in range(0, len(order), batch_size):
            ids = order[start:start+batch_size]
            batch = start // batch_size
            classes = preferred[self.labels[ids]]
            for i in np.flatnonzero((full[classes] == batch).any(1)):
                candidates = self.ordered[self.labels[ids[i]]]
                eligible = candidates[full[candidates] != batch]
                if self.mode == "near":
                    eligible = eligible[:20]
                classes[i] = rng.choice(eligible, 16, replace=False)
            rows = self.rows[classes]
            draw = rng.random(rows.shape, dtype=np.float32)
            draw[batch_id[rows] == batch] = -1
            result[ids] = np.take_along_axis(rows, draw.argmax(2)[:, :, None], axis=2)[:, :, 0]
        return result
