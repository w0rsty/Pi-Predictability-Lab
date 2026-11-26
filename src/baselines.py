# Классические бейзлайны (Uniform, цепи Маркова k=1..3, LightGBM) и векторизованный PPM-C
import math
import time
from typing import Any

import lightgbm as lgb
import numpy as np

from src.dataset import ChronologicalSplit, build_tabular_windows


def compute_prediction_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Считает честные метрики: Overall Acc, Targeted Acc (по mask==1), Cross-Entropy, PPL и энтропию."""
    probs_safe = np.clip(probs.astype(np.float64, copy=False), 1e-12, 1.0)
    # На всякий случай нормируем строки, чтобы сумма вероятностей была равна 1
    probs_safe = probs_safe / probs_safe.sum(axis=1, keepdims=True)

    preds = np.argmax(probs_safe, axis=1)
    correct = (preds == targets)
    overall_acc = float(np.mean(correct))

    n = len(targets)
    target_probs = probs_safe[np.arange(n), targets]
    cross_entropy = float(-np.mean(np.log(target_probs)))
    perplexity = float(math.exp(min(cross_entropy, 50.0)))
    pred_entropy = float(-np.mean(np.sum(probs_safe * np.log(probs_safe), axis=1)))

    targeted_acc: float | None = None
    targeted_ce: float | None = None
    mask_count = 0
    if mask is not None:
        mask_bool = mask == 1
        mask_count = int(np.sum(mask_bool))
        if mask_count > 0:
            targeted_acc = float(np.mean(correct[mask_bool]))
            targeted_ce = float(-np.mean(np.log(target_probs[mask_bool])))

    return {
        "overall_acc": overall_acc,
        "targeted_acc": targeted_acc,
        "cross_entropy": cross_entropy,
        "targeted_ce": targeted_ce,
        "perplexity": perplexity,
        "pred_entropy": pred_entropy,
        "num_eval_samples": n,
        "num_masked_samples": mask_count,
    }


def evaluate_uniform_baseline(
    split: ChronologicalSplit,
    radix: int,
    window_size: int,
) -> dict[str, Any]:
    """Оценка наивного равномерного угадывания P(d) = 1/radix на тестовом срезе."""
    t0 = time.perf_counter()
    targets = split.test_digits[window_size:]
    mask = split.test_mask[window_size:] if split.test_mask is not None else None
    n = len(targets)

    # При равномерном распределении argmax=0, но вероятности строго 1/radix
    probs = np.full((n, radix), 1.0 / radix, dtype=np.float32)
    metrics = compute_prediction_metrics(probs, targets, mask)
    metrics["model"] = "Uniform"
    metrics["train_time_sec"] = 0.0
    metrics["eval_time_sec"] = round(time.perf_counter() - t0, 4)
    return metrics


def _encode_markov_states(digits: np.ndarray, order: int, radix: int) -> np.ndarray:
    """Кодирует окна длины order в целые индексы состояний 0..radix^order - 1."""
    view = np.lib.stride_tricks.sliding_window_view(digits[:-1], window_shape=order)
    powers = (radix ** np.arange(order - 1, -1, -1, dtype=np.int64))
    return view.astype(np.int64) @ powers


def fit_and_evaluate_markov(
    split: ChronologicalSplit,
    radix: int,
    order: int,
    window_size: int,
    smoothing: float = 1e-3,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Обучает цепь Маркова порядка k на Train и оценивает на Test. Для k=1 возвращает матрицу переходов."""
    t0 = time.perf_counter()
    num_states = radix**order

    # Считаем переходы строго на обучающем срезе
    train_states = _encode_markov_states(split.train_digits, order=order, radix=radix)
    train_targets = split.train_digits[order:].astype(np.int64)
    flat_idx = train_states * radix + train_targets
    counts = np.bincount(flat_idx, minlength=num_states * radix).reshape(num_states, radix).astype(np.float64)

    # Сглаживание Лапласа защищает от log(0), если редкое состояние не встречалось в Train
    probs_table = (counts + smoothing) / (counts.sum(axis=1, keepdims=True) + smoothing * radix)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    # Чтобы честно сравнивать все модели на одних и тех же тестовых позициях, берём индексы >= window_size
    test_offset = window_size - order
    if test_offset < 0:
        raise ValueError("window_size должен быть >= порядка цепи Маркова")

    test_slice = split.test_digits[test_offset:]
    test_states = _encode_markov_states(test_slice, order=order, radix=radix)
    test_targets = split.test_digits[window_size:]
    test_mask = split.test_mask[window_size:] if split.test_mask is not None else None

    pred_probs = probs_table[test_states]
    metrics = compute_prediction_metrics(pred_probs, test_targets, test_mask)
    metrics["model"] = f"Markov_k{order}"
    metrics["train_time_sec"] = round(train_time, 4)
    metrics["eval_time_sec"] = round(time.perf_counter() - t1, 4)

    transition_matrix = probs_table if order == 1 else None
    return metrics, transition_matrix


def compute_empirical_transition_matrix(digits: np.ndarray, radix: int) -> np.ndarray:
    """Считает эмпирическую матрицу условных переходов P(x_{t+1} | x_t) размера radix x radix."""
    curr = digits[:-1].astype(np.int64)
    nxt = digits[1:].astype(np.int64)
    counts = np.bincount(curr * radix + nxt, minlength=radix * radix).reshape(radix, radix).astype(np.float64)
    row_sums = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return counts / row_sums


def fit_and_evaluate_lightgbm(
    split: ChronologicalSplit,
    radix: int,
    window_size: int,
    seed: int = 42,
    max_train_samples: int = 200_000,
    n_estimators: int = 150,
    learning_rate: float = 0.08,
) -> dict[str, Any]:
    """Обучает LightGBM на окне длины window_size с ранней остановкой по Val и оценивает на Test."""
    t0 = time.perf_counter()
    x_train, y_train, _ = build_tabular_windows(
        split.train_digits, split.train_mask, window_size=window_size, max_samples=max_train_samples
    )
    x_val, y_val, _ = build_tabular_windows(
        split.val_digits, split.val_mask, window_size=window_size
    )
    x_test, y_test, m_test = build_tabular_windows(
        split.test_digits, split.test_mask, window_size=window_size
    )

    # Приводим к float32 — LightGBM работает быстро и с max_bin=32 точно нарезает целые цифры 0..15
    x_train_f = np.ascontiguousarray(x_train, dtype=np.float32)
    x_val_f = np.ascontiguousarray(x_val, dtype=np.float32)
    x_test_f = np.ascontiguousarray(x_test, dtype=np.float32)

    objective = "binary" if radix == 2 else "multiclass"
    model_kwargs: dict[str, Any] = {
        "objective": objective,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": 31,
        "max_bin": 32,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if radix > 2:
        model_kwargs["num_class"] = radix

    clf = lgb.LGBMClassifier(**model_kwargs)
    clf.fit(
        x_train_f,
        y_train,
        eval_set=[(x_val_f, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
    )
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_probs = clf.predict_proba(x_test_f)
    # Если редкий класс отсутствует в y_train, расширяем матрицу вероятностей до полного числа классов radix
    if pred_probs.shape[1] < radix:
        full_probs = np.zeros((len(x_test_f), radix), dtype=np.float64)
        for col_idx, cls_id in enumerate(clf.classes_):
            full_probs[:, int(cls_id)] = pred_probs[:, col_idx]
        pred_probs = full_probs

    metrics = compute_prediction_metrics(pred_probs, y_test, m_test)
    metrics["model"] = "LightGBM"
    metrics["best_iteration"] = int(clf.best_iteration_ or n_estimators)
    metrics["train_samples_used"] = int(len(y_train))
    metrics["train_time_sec"] = round(train_time, 4)
    metrics["eval_time_sec"] = round(time.perf_counter() - t1, 4)
    return metrics


class PPMCompressor:
    """Суффиксный компрессор PPM-C до порядка max_order с escape-откатом и exclusion.

    На Base-10/Base-16 при order=8 пространство суффиксов разрежено, поэтому на шуме битрейт улетает
    выше предела Шеннона (3.774 / 4.481 бит), зато в Base-2 (2^8=256 состояний) держит честные 1.0002 бита.
    """

    def __init__(self, radix: int, max_order: int = 8, dense_limit: int = 200_000) -> None:
        # max_order = 4  # на order=4 в Base-10 таблица не так раздувается, но для эталона оставил 8
        self.radix = int(radix)
        self.max_order = int(max_order)
        self.dense_limit = int(dense_limit)
        self.order0_counts: np.ndarray | None = None
        self.dense_tables: dict[int, np.ndarray] = {}
        self.sparse_tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _encode_contexts(self, digits: np.ndarray, order: int, end_offset: int = 1) -> np.ndarray:
        """Кодирует окна длины order в целочисленные ключи int64."""
        src = digits[:-end_offset] if end_offset > 0 else digits
        view = np.lib.stride_tricks.sliding_window_view(src, window_shape=order)
        powers = self.radix ** np.arange(order - 1, -1, -1, dtype=np.int64)
        return view.astype(np.int64) @ powers

    def fit(self, train_digits: np.ndarray) -> "PPMCompressor":
        """Строит таблицы счётчиков для всех длин контекста k = 0..max_order на Train."""
        self.order0_counts = np.bincount(train_digits.astype(np.int64), minlength=self.radix).astype(np.int32)
        self.dense_tables.clear()
        self.sparse_tables.clear()

        for order in range(1, self.max_order + 1):
            codes = self._encode_contexts(train_digits, order=order, end_offset=1)
            targets = train_digits[order:].astype(np.int64)
            num_states = self.radix**order

            if num_states <= self.dense_limit:
                flat_idx = codes * self.radix + targets
                counts = (
                    np.bincount(flat_idx, minlength=num_states * self.radix)
                    .reshape(num_states, self.radix)
                    .astype(np.int32)
                )
                self.dense_tables[order] = counts
            else:
                # Для длинных контекстов храним только реально встретившиеся суффиксы (O(N_train))
                pairs = codes * self.radix + targets
                uniq_pairs, pair_counts = np.unique(pairs, return_counts=True)
                pair_ctx = uniq_pairs // self.radix
                pair_sym = uniq_pairs % self.radix
                uniq_ctx, inv_idx = np.unique(pair_ctx, return_inverse=True)
                counts_mat = np.zeros((len(uniq_ctx), self.radix), dtype=np.int32)
                counts_mat[inv_idx, pair_sym] = pair_counts.astype(np.int32)
                self.sparse_tables[order] = (uniq_ctx, counts_mat)

        return self

    def _lookup_order_counts(self, test_digits: np.ndarray, order: int, window_size: int) -> np.ndarray:
        """Возвращает матрицу счётчиков (N_test, radix) для позиций >= window_size."""
        n_eval = len(test_digits) - window_size
        if order == 0:
            assert self.order0_counts is not None
            return np.broadcast_to(self.order0_counts[None, :], (n_eval, self.radix)).copy()

        start_offset = window_size - order
        if start_offset < 0:
            raise ValueError(f"window_size ({window_size}) должен быть >= max_order ({self.max_order})")

        sliced = test_digits[start_offset:]
        codes = self._encode_contexts(sliced, order=order, end_offset=1)
        if order in self.dense_tables:
            return self.dense_tables[order][codes]

        uniq_ctx, counts_mat = self.sparse_tables[order]
        if len(uniq_ctx) == 0:
            return np.zeros((n_eval, self.radix), dtype=np.int32)
        idx = np.searchsorted(uniq_ctx, codes)
        safe_idx = np.minimum(idx, len(uniq_ctx) - 1)
        hit = (idx < len(uniq_ctx)) & (uniq_ctx[safe_idx] == codes)
        out = np.zeros((n_eval, self.radix), dtype=np.int32)
        out[hit] = counts_mat[safe_idx[hit]]
        return out

    def predict_proba(self, test_digits: np.ndarray, window_size: int) -> np.ndarray:
        """Считает вероятности PPM-C с исключением (exclusion) и escape от max_order до -1."""
        n_eval = len(test_digits) - window_size
        probs = np.zeros((n_eval, self.radix), dtype=np.float64)
        rem_mass = np.ones((n_eval, 1), dtype=np.float64)
        excluded = np.zeros((n_eval, self.radix), dtype=bool)

        for order in range(self.max_order, -1, -1):
            c_raw = self._lookup_order_counts(test_digits, order=order, window_size=window_size)
            c_eff = np.where(excluded, 0, c_raw).astype(np.float64)
            n_eff = c_eff.sum(axis=1, keepdims=True)
            t_eff = (c_eff > 0).sum(axis=1, keepdims=True).astype(np.float64)

            has_ctx = (n_eff > 0).squeeze(1)
            if np.any(has_ctx):
                # Формула PPM-C: наблюдаемый счёт делим на (N_eff + T_eff), а T_eff уходит в escape на более короткий контекст.
                # При radix=10/16 и order=8 контексты в 1M-ряду почти не повторяются, поэтому модель часто откатывается вниз.
                denom = n_eff[has_ctx] + t_eff[has_ctx]
                p_k = c_eff[has_ctx] / denom
                e_k = t_eff[has_ctx] / denom
                probs[has_ctx] += rem_mass[has_ctx] * p_k
                rem_mass[has_ctx] *= e_k
                # Exclusion: символы, уже оцененные на длинном суффиксе, не учитываются заново на коротком
                excluded[has_ctx] |= c_raw[has_ctx] > 0

        # Уровень -1: равномерный остаток по ещё не встретившимся символам (или 1/radix, если видели все)
        unseen = ~excluded
        n_unseen = unseen.sum(axis=1, keepdims=True)
        uniform_fallback = np.where(
            n_unseen > 0,
            unseen.astype(np.float64) / np.maximum(n_unseen, 1),
            1.0 / self.radix,
        )
        probs += rem_mass * uniform_fallback
        return probs


def fit_and_evaluate_ppm(
    split: ChronologicalSplit,
    radix: int,
    window_size: int,
    max_order: int = 8,
) -> tuple[dict[str, Any], np.ndarray]:
    """Обучает PPM-C на Train, оценивает на Test и возвращает (метрики, вероятности)."""
    t0 = time.perf_counter()
    ppm = PPMCompressor(radix=radix, max_order=max_order).fit(split.train_digits)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_probs = ppm.predict_proba(split.test_digits, window_size=window_size)
    targets = split.test_digits[window_size:]
    mask = split.test_mask[window_size:] if split.test_mask is not None else None

    metrics = compute_prediction_metrics(pred_probs, targets, mask)
    metrics["model"] = f"PPM_o{max_order}"
    metrics["bits_per_symbol"] = float(metrics["cross_entropy"] / math.log(2.0))
    metrics["targeted_bits_per_symbol"] = (
        float(metrics["targeted_ce"] / math.log(2.0)) if metrics["targeted_ce"] is not None else None
    )
    metrics["train_time_sec"] = round(train_time, 4)
    metrics["eval_time_sec"] = round(time.perf_counter() - t1, 4)
    return metrics, pred_probs
