import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
import numpy as np
import scipy.sparse as sp
from models import TKBCModel


DATA_PATH = 'pre_data/'


class TemporalDataset:
    """
    Temporal Dataset Loader for Time-aware Knowledge Graph Completion (e.g., ICEWS14, ICEWS05-15).

    支持：
    - 普通时间戳 (t)
    - 时间区间 (t_start, t_end)
    - 双向关系扩展
    - time-aware filtered 评估
    """

    def __init__(self, name: str):
        """
        初始化并加载数据集
        Args:
            name: 数据集名称（例如 'ICEWS14'）
        """
        self.root = Path(DATA_PATH) / name
        self.data: Dict[str, np.ndarray] = {}
        self.interval: bool = False
        self.events: Optional[List] = None
        self.time_dict: Optional[Dict] = None
        self.time_diffs: Optional[torch.Tensor] = None

        # === 加载 train/valid/test ===
        for split in ['train', 'valid', 'test']:
            file_path = self.root / f"{split}.pickle"
            with open(file_path, 'rb') as f:
                self.data[split] = pickle.load(f)

        # === 统计实体/关系/时间戳数 ===
        max_values = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(max_values[0], max_values[2]) + 1)
        self.n_predicates = int(max_values[1] + 1) * 2  # 双向关系
        self.n_timestamps = (
            int(max(max_values[3] + 1, max_values[4] + 1))
            if max_values.shape[0] > 4
            else int(max_values[3] + 1)
        )

        # === 时间区间判断 ===
        if self.data['valid'].shape[1] > 4:
            self.interval = True
            ts_id_path = self.root / 'ts_id.pickle'
            if ts_id_path.exists():
                with open(ts_id_path, 'rb') as f:
                    self.time_dict = pickle.load(f)

        # === 时间间隔信息 ===
        ts_diff_path = self.root / 'ts_diffs.pickle'
        if ts_diff_path.exists():
            with open(ts_diff_path, 'rb') as f:
                self.time_diffs = torch.from_numpy(pickle.load(f)).float().cuda()
        else:
            print("[INFO] Assume all timestamps are regularly spaced.")
            self.time_diffs = None

        # === 事件与时间戳加载（用于 time-aware 评估） ===
        event_list_path = self.root / 'event_list_all.pickle'
        ts_id_path_alt = self.root / 'ts_id'
        if event_list_path.exists() and ts_id_path_alt.exists():
            with open(event_list_path, 'rb') as f1, open(ts_id_path_alt, 'rb') as f2:
                self.events = pickle.load(f1)
                time_dict = pickle.load(f2)
                self.timestamps = sorted(time_dict.keys())
        else:
            print("[INFO] No event list found — use standard evaluation.")
            self.events = None
            self.timestamps = None

        # === 加载过滤字典 ===
        to_skip_path = self.root / 'to_skip.pickle'
        if to_skip_path.exists():
            with open(to_skip_path, 'rb') as f:
                self.to_skip: Dict[str, Dict[Tuple[int, int, int], List[int]]] = pickle.load(f)
        else:
            raise FileNotFoundError(f"Missing filtering dictionary: {to_skip_path}")

    # ------------------------------------------------------------------------
    # Dataset access
    # ------------------------------------------------------------------------
    def get_examples(self, split: str) -> np.ndarray:
        """返回指定数据划分（train/test/valid）的样本"""
        return self.data[split]

    def get_train(self) -> np.ndarray:
        """
        返回包含双向关系的训练样本。
        - 为每个 (h, r, t) 添加 (t, r', h)
        - 关系索引加上 n_predicates // 2 表示反向关系
        """
        train = np.copy(self.data['train'])
        reversed_train = np.copy(train)
        reversed_train[:, 0], reversed_train[:, 2] = train[:, 2], train[:, 0]
        reversed_train[:, 1] += self.n_predicates // 2
        return np.vstack((train, reversed_train))

    def get_shape(self) -> Tuple[int, int, int, int]:
        """返回 (n_entities, n_predicates, n_entities, n_timestamps)"""
        return self.n_entities, self.n_predicates, self.n_entities, self.n_timestamps

    # ------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------
    def eval(
        self,
        model: TKBCModel,
        split: str,
        n_queries: int = -1,
        missing_eval: str = 'both',
        at: Tuple[int] = (1, 3, 10),
        result_ranks:bool = False
    ):
        """
        Evaluate a model on a given split (train/valid/test)

        Args:
            model: TKBCModel 模型实例
            split: 数据集划分
            n_queries: 采样数量（-1 表示全量）
            missing_eval: {'lhs', 'rhs', 'both'}
            at: Top-K 指标列表
        Returns:
            (mean_reciprocal_rank, hits_at)
        """
        examples = self.get_examples(split)
        missing = ['rhs'] if missing_eval == 'rhs' else (
            ['lhs'] if missing_eval == 'lhs' else ['rhs', 'lhs']
        )

        mean_reciprocal_rank, hits_at = {}, {}

        # 时间区间模式
        if self.interval:
            for side in missing:
                q = np.copy(examples)
                if side == 'lhs':  # 左侧预测
                    q[:, [0, 2]] = q[:, [2, 0]]
                    q[:, 1] = q[:, 1].astype('uint64') + self.n_predicates // 2

                ranks = model.get_ranking(
                    q, self.to_skip[side], batch_size=500, year2id=self.time_dict
                )
                mean_reciprocal_rank[side] = torch.mean(1. / ranks).item()
                hits_at[side] = torch.FloatTensor([
                    torch.mean((ranks <= k).float()).item() for k in at
                ])

        # 普通时间戳模式
        else:
            examples_torch = torch.from_numpy(examples.astype('int64')).cuda()
            for side in missing:
                q = examples_torch.clone()
                if n_queries > 0:
                    perm = torch.randperm(len(q))[:n_queries]
                    q = q[perm]

                if side == 'lhs':  # 左侧预测
                    q[:, [0, 2]] = q[:, [2, 0]]
                    q[:, 1] += self.n_predicates // 2

                ranks = model.get_ranking(q, self.to_skip[side], batch_size=500)
                mean_reciprocal_rank[side] = torch.mean(1. / ranks).item()
                hits_at[side] = torch.FloatTensor([
                    torch.mean((ranks <= k).float()).item() for k in at
                ])

        if result_ranks:
            return mean_reciprocal_rank, hits_at, ranks
        else:
            return mean_reciprocal_rank, hits_at


class HistoryFreqFilter:
    """
    历史频率过滤器
    - 从 data/{dataset}/history_seq/h_r_history_train_valid.npz 加载
    - 支持 top-k 或比例阈值过滤
    """

    def __init__(self, dataset: str, freq_dir: str = None, threshold: float = 0.8, topk: int = None):
        self.dataset = dataset
        self.freq_dir = freq_dir or os.path.join("pre_data", dataset, "history_seq")
        self.threshold = float(threshold)
        self.topk = topk

        path = os.path.join(self.freq_dir, "h_r_history_train_valid.npz")
        if os.path.exists(path):
            self.base_mat = sp.load_npz(path).tocsr()
            print(f"[HistoryFreqFilter] Loaded {path}")
            print(f"  shape = {self.base_mat.shape}")
            print(f"  非零数 = {self.base_mat.nnz}")
            print(f"  稀疏比例 = {self.base_mat.nnz / (self.base_mat.shape[0] * self.base_mat.shape[1]):.8f}")
        else:
            self.base_mat = None
            print(f"[HistoryFreqFilter] WARN: {path} not found, disabled.")

    def score_batch(self, s, r, normalize=False, k=1.0):
        """
        返回每个 (s, r) 对应的历史频率分布向量 [B, num_entities]
        """
        if self.base_mat is None:
            return None

        num_entities = self.base_mat.shape[1]
        num_rels = (self.base_mat.shape[0] // num_entities) // 2
        outputs = []

        for si, ri in zip(s.tolist(), r.tolist()):
            ri = ri % (2 * num_rels)
            idx = si * (2 * num_rels) + ri
            if idx >= self.base_mat.shape[0]:
                idx = self.base_mat.shape[0] - 1
            row = self.base_mat.getrow(idx).toarray().flatten()
            if normalize and row.sum() > 0:
                row = row / row.sum()
            outputs.append(row * k)
        return torch.tensor(outputs, dtype=torch.float32)

    def filter_candidates(self, s: int, r: int, num_entities: int, chunk_start=0, chunk_size=None, device="cpu"):
        """
        生成候选实体掩码 (True = 保留)
        """
        if self.base_mat is None:
            return torch.ones(num_entities if chunk_size is None else chunk_size, dtype=torch.bool, device=device)

        num_rels = (self.base_mat.shape[0] // num_entities) // 2
        idx = s * (2 * num_rels) + (r % (2 * num_rels))
        if idx >= self.base_mat.shape[0]:
            idx = self.base_mat.shape[0] - 1

        row = self.base_mat.getrow(idx).toarray().flatten()
        if np.sum(row) == 0:
            return torch.ones(num_entities if chunk_size is None else chunk_size, dtype=torch.bool, device=device)

        k = self.topk or int(len(row) * (1.0 - self.threshold))
        k = max(1, min(k, len(row)))
        top_idx = np.argpartition(-row, k - 1)[:k]

        mask = torch.zeros(num_entities, dtype=torch.bool)
        mask[top_idx] = True
        if chunk_size is not None:
            chunk_end = min(chunk_start + chunk_size, num_entities)
            mask = mask[chunk_start:chunk_end]
        return mask.to(device)



