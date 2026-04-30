"""
Wrapper (evolutionary) feature selectors – optional EA extra.

Install::

    pip install "git+https://github.com/xwdshiwo/clinifs.git@v0.1.0"

Included methods
----------------
GASelector   : Genetic Algorithm (custom implementation)
BPSOSelector : Binary Particle Swarm Optimisation (custom implementation)

**MEL is intentionally excluded** from this package.  If you have MEL scores
from an external run, pass them to ``RankAggregateFilter`` via the
``extra_scores={'mel': <array>}`` argument.

Runtime note
------------
Each fit call runs a population-based search (30 pop × 20 gen by default)
with internal 3-fold CV per individual.  Expect 30–120 s on microarray
data after the ANOVA pre-filter.  These methods are **not suitable** for
real-time deployment; use RankAggregateFilter for online inference.
"""
import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted

from clinifs._prefilter import anova_prefilter


# ──────────────────────────────────────────────────────────────────────────────
# Internal GA implementation (ported from benchmark src/methods/ga_custom.py)
# ──────────────────────────────────────────────────────────────────────────────

class _GAEngine:
    """Lightweight GA with sparsity penalty (no external dependencies)."""

    def __init__(self, alpha=0.5, beta=0.5, pop_size=30, n_gen=20,
                 max_features=None, mutation_rate=0.05, crossover_rate=0.8,
                 tournament_size=3, elite_size=2,
                 random_state=42, verbose=False, cv=3):
        self.alpha = alpha
        self.beta = beta
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.max_features = max_features
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.random_state = random_state
        self.verbose = verbose
        self.cv = cv

    def _fitness(self, X, y, chrom):
        if chrom.sum() == 0:
            return 1.0
        X_sel = X[:, chrom.astype(bool)]
        clf = LogisticRegression(
            C=1, solver="liblinear", max_iter=300, random_state=42
        )
        try:
            auc = float(np.mean(
                cross_val_score(clf, X_sel, y, cv=self.cv,
                                scoring="roc_auc", n_jobs=1)
            ))
        except Exception:
            auc = 0.5
        return self.alpha * (1 - auc) + self.beta * (chrom.sum() / len(chrom))

    def _tournament(self, pop, fitness, rng):
        idx = rng.choice(len(pop), self.tournament_size, replace=False)
        return pop[idx[np.argmin(fitness[idx])]]

    def _crossover(self, p1, p2, rng):
        if rng.random() > self.crossover_rate:
            return p1.copy()
        pt = rng.randint(1, len(p1))
        child = np.concatenate([p1[:pt], p2[pt:]])
        if self.max_features and child.sum() > self.max_features:
            on_idx = np.where(child)[0]
            off = rng.choice(on_idx, int(child.sum()) - self.max_features,
                             replace=False)
            child[off] = 0
        return child

    def _mutate(self, chrom, rng):
        mask = rng.random(len(chrom)) < self.mutation_rate
        chrom = chrom ^ mask.astype(int)
        if self.max_features and chrom.sum() > self.max_features:
            on_idx = np.where(chrom)[0]
            off = rng.choice(on_idx, int(chrom.sum()) - self.max_features,
                             replace=False)
            chrom[off] = 0
        return chrom

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_feat = X.shape[1]
        kmax = self.max_features or n_feat

        pop = (rng.random((self.pop_size, n_feat)) < 0.5).astype(int)
        if kmax < n_feat:
            for i in range(self.pop_size):
                on = np.where(pop[i])[0]
                if len(on) > kmax:
                    pop[i][rng.choice(on, len(on) - kmax, replace=False)] = 0

        fitness = np.array([self._fitness(X, y, pop[i]) for i in range(self.pop_size)])
        best_idx = int(np.argmin(fitness))
        self.best_features_ = np.where(pop[best_idx])[0]
        self.best_fitness_ = fitness[best_idx]
        self.fitness_curve_ = [self.best_fitness_]
        self.feature_frequencies_ = None

        for gen in range(self.n_gen):
            order = np.argsort(fitness)
            new_pop = pop[order[: self.elite_size]].copy()
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fitness, rng)
                p2 = self._tournament(pop, fitness, rng)
                child = self._mutate(self._crossover(p1, p2, rng), rng)
                new_pop = np.vstack([new_pop, child])
            pop = new_pop
            fitness = np.array([self._fitness(X, y, pop[i])
                                for i in range(self.pop_size)])
            best_idx = int(np.argmin(fitness))
            if fitness[best_idx] < self.best_fitness_:
                self.best_fitness_ = fitness[best_idx]
                self.best_features_ = np.where(pop[best_idx])[0]
            self.fitness_curve_.append(self.best_fitness_)
            if self.verbose:
                print(f"  GA gen {gen+1}/{self.n_gen}  "
                      f"best_fitness={self.best_fitness_:.4f}  "
                      f"n_feat={len(self.best_features_)}")

        freqs = pop.mean(axis=0)
        self.feature_frequencies_ = freqs
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Internal BPSO implementation (ported from src/algorithms/binary_pso.py)
# ──────────────────────────────────────────────────────────────────────────────

class _BPSOEngine:
    """Binary PSO (Kennedy & Eberhart 1997) with sparsity penalty."""

    def __init__(self, n_particles=20, max_iter=30, w=0.9, c1=2.0, c2=2.0,
                 alpha=0.5, beta=0.5, cv=3, random_state=42, verbose=False):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.beta = beta
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def _sigmoid(v):
        return 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))

    def _fitness(self, X, y, pos):
        selected = np.where(pos.astype(bool))[0]
        if len(selected) == 0:
            return 1.0
        X_sel = X[:, selected]
        clf = LogisticRegression(
            C=1, solver="liblinear", max_iter=300, random_state=42
        )
        try:
            auc = float(np.mean(
                cross_val_score(clf, X_sel, y, cv=self.cv,
                                scoring="roc_auc", n_jobs=1)
            ))
        except Exception:
            auc = 0.5
        return self.alpha * (1 - auc) + self.beta * (len(selected) / X.shape[1])

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_feat = X.shape[1]

        pos = rng.randint(0, 2, (self.n_particles, n_feat)).astype(float)
        vel = rng.uniform(-1, 1, (self.n_particles, n_feat))
        fitness = np.array([self._fitness(X, y, pos[i]) for i in range(self.n_particles)])

        pbest_pos = pos.copy()
        pbest_fit = fitness.copy()
        gbest_idx = int(np.argmin(pbest_fit))
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        for it in range(self.max_iter):
            r1 = rng.random((self.n_particles, n_feat))
            r2 = rng.random((self.n_particles, n_feat))
            vel = (self.w * vel
                   + self.c1 * r1 * (pbest_pos - pos)
                   + self.c2 * r2 * (gbest_pos - pos))
            prob = self._sigmoid(vel)
            pos = (rng.random((self.n_particles, n_feat)) < prob).astype(float)

            fitness = np.array([self._fitness(X, y, pos[i])
                                for i in range(self.n_particles)])
            improved = fitness < pbest_fit
            pbest_pos[improved] = pos[improved]
            pbest_fit[improved] = fitness[improved]

            best_i = int(np.argmin(pbest_fit))
            if pbest_fit[best_i] < gbest_fit:
                gbest_fit = pbest_fit[best_i]
                gbest_pos = pbest_pos[best_i].copy()

            if self.verbose:
                print(f"  BPSO iter {it+1}/{self.max_iter}  "
                      f"gbest_fit={gbest_fit:.4f}  "
                      f"n_feat={int(gbest_pos.sum())}")

        self.best_features_ = np.where(gbest_pos.astype(bool))[0]
        self.best_fitness_ = gbest_fit
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Public sklearn-compatible selectors
# ──────────────────────────────────────────────────────────────────────────────

class GASelector(BaseEstimator, TransformerMixin):
    """
    Evolutionary wrapper feature selection via Genetic Algorithm.

    Fitness = alpha * (1 - AUC_cv) + beta * (|S| / p)  [minimise]

    Parameters
    ----------
    k : int
        Number of features to select.
    alpha, beta : float
        Weights for AUC loss and sparsity penalty (sum to 1.0 recommended).
    pop_size : int
        GA population size.
    n_gen : int
        Number of generations.
    prefilter : int
        Reduce to top-n ANOVA features before the GA search.

    Notes
    -----
    Based on the benchmark implementation used in the paper.
    Benchmarked AUC gap vs Filter on Easy tier: < 0.5 pp (negligible).
    Hard tier: mean AUC 0.648 vs Filter 0.618.
    """

    def __init__(self, k: int = 20, alpha: float = 0.5, beta: float = 0.5,
                 pop_size: int = 30, n_gen: int = 20,
                 prefilter: int = 2000, random_state: int = 42,
                 verbose: bool = False):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.prefilter = prefilter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        warnings.warn(
            "GASelector runtime is O(pop_size × n_gen × CV_folds × n_samples). "
            "Expect 30–120 s on microarray data. "
            "For real-time or web deployment use RankAggregateFilter instead.",
            UserWarning, stacklevel=2,
        )
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]

        X_sub, pre_idx, _ = anova_prefilter(X, y, self.prefilter)
        engine = _GAEngine(
            alpha=self.alpha, beta=self.beta,
            pop_size=self.pop_size, n_gen=self.n_gen,
            max_features=self.k,
            random_state=self.random_state, verbose=self.verbose
        )
        engine.fit(X_sub, y)
        sub_idx = engine.best_features_

        if len(sub_idx) > self.k:
            freqs = engine.feature_frequencies_[sub_idx]
            sub_idx = sub_idx[np.argsort(-freqs)[: self.k]]
        elif len(sub_idx) < self.k:
            from sklearn.feature_selection import f_classif as _fc
            sc, _ = _fc(X_sub, y)
            sc = np.nan_to_num(sc, nan=0.0)
            ranked = np.argsort(-sc)
            extra = [i for i in ranked if i not in set(sub_idx)][
                : self.k - len(sub_idx)
            ]
            sub_idx = np.concatenate([sub_idx, np.array(extra, dtype=int)])

        self.selected_indices_ = pre_idx[sub_idx]
        self.scores_ = None
        self.ea_info_ = {
            "fitness_curve": engine.fitness_curve_,
            "best_fitness": float(engine.best_fitness_),
            "n_gen": self.n_gen,
        }
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]


class BPSOSelector(BaseEstimator, TransformerMixin):
    """
    Evolutionary wrapper feature selection via Binary PSO.

    .. warning::
        Benchmark results show BPSO's sparsity penalty is ineffective under
        default (alpha=0.5, beta=0.5): actual panel sizes reach 871–980
        regardless of beta.  Treat BPSO outputs as uncontrolled panel size
        and post-truncate to k by ANOVA score (done automatically here).

    Parameters
    ----------
    k : int
        Number of features to select (post-truncation applied).
    alpha, beta : float
        Fitness weights; beta > 0.5 recommended to encourage sparsity.
    """

    def __init__(self, k: int = 20, alpha: float = 0.5, beta: float = 0.5,
                 n_particles: int = 20, max_iter: int = 30,
                 prefilter: int = 2000, random_state: int = 42,
                 verbose: bool = False):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.prefilter = prefilter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        warnings.warn(
            "BPSOSelector: sparsity penalty is known to be ineffective under "
            "default parameters (see benchmark paper §3.7). "
            "Panel is post-truncated to k by ANOVA score.",
            UserWarning, stacklevel=2,
        )
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]

        X_sub, pre_idx, _ = anova_prefilter(X, y, self.prefilter)
        engine = _BPSOEngine(
            n_particles=self.n_particles, max_iter=self.max_iter,
            alpha=self.alpha, beta=self.beta,
            random_state=self.random_state, verbose=self.verbose
        )
        engine.fit(X_sub, y)
        sub_idx = engine.best_features_

        if len(sub_idx) > self.k:
            from sklearn.feature_selection import f_classif as _fc
            sc, _ = _fc(X_sub[:, sub_idx], y)
            sc = np.nan_to_num(sc, nan=0.0)
            sub_idx = sub_idx[np.argsort(-sc)[: self.k]]
        elif len(sub_idx) < self.k:
            from sklearn.feature_selection import f_classif as _fc
            sc, _ = _fc(X_sub, y)
            sc = np.nan_to_num(sc, nan=0.0)
            ranked = np.argsort(-sc)
            extra = [i for i in ranked if i not in set(sub_idx)][
                : self.k - len(sub_idx)
            ]
            sub_idx = np.concatenate([sub_idx, np.array(extra, dtype=int)])

        self.selected_indices_ = pre_idx[sub_idx]
        self.scores_ = None
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(X)[:, self.selected_indices_]
