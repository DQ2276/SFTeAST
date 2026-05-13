import os
import pickle
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_pickle_data(base_path):
    """加载 train / valid / test 三个 pickle 文件"""
    data = []
    for name in ['train', 'valid', 'test']:
        file_path = os.path.join(base_path, f"{name}.pickle")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                arr = pickle.load(f)
                data.append(arr)
    return np.concatenate(data, axis=0)


def aggregate_time(timestamps, dataset):
    """将时间戳聚合为较粗时间粒度，降低稀疏性"""
    if "ICEWS14" in dataset:
        return timestamps // 7      # 每 7 天一个时间片
    elif "ICEWS05-15" in dataset:
        return timestamps // 14     # 每 14 天一个时间片
    elif "GDELT" in dataset:
        return timestamps // 30     # 每 30 天一个时间片
    else:
        return timestamps


def generate_history_npz(dataset):
    """
    根据 prepare_dataset 生成的数据文件，构建历史频率矩阵
    """
    data_dir = os.path.join("pre_data", dataset)
    save_dir = os.path.join(data_dir, "history_seq")
    mkdirs(save_dir)

    # 读取映射文件
    with open(os.path.join(data_dir, "ent_id"), "r", encoding="utf-8") as f:
        num_e = len(f.readlines())
    with open(os.path.join(data_dir, "rel_id"), "r", encoding="utf-8") as f:
        num_r = len(f.readlines())

    print(f"[{dataset}] num_entities={num_e}, num_relations={num_r}")

    # 加载全部四元组
    all_data = load_pickle_data(data_dir)
    print(f"Loaded quadruples: {len(all_data)}")

    # 时间聚合
    all_data[:, 3] = aggregate_time(all_data[:, 3], dataset)

    num_r_2 = num_r * 2

    # 生成正向 + 反向关系索引
    heads = all_data[:, 0]
    rels = all_data[:, 1]
    tails = all_data[:, 2]

    rows = np.concatenate([heads * num_r_2 + rels, tails * num_r_2 + (rels + num_r)])
    cols = np.concatenate([tails, heads])
    data = np.ones(len(rows), dtype=np.float32)

    # 构建稀疏矩阵
    mat = sp.csr_matrix((data, (rows, cols)), shape=(num_e * num_r_2, num_e))

    # 保存
    save_path = os.path.join(save_dir, "h_r_history_train_valid.npz")
    sp.save_npz(save_path, mat)

    print(f"[OK] Saved {save_path}")
    print(f"  shape: {mat.shape}")
    print(f"  非零数: {mat.nnz}")
    print(f"  稀疏比例: {mat.nnz / (mat.shape[0] * mat.shape[1]):.8e}")


if __name__ == "__main__":
    for dataset in ["ICEWS14", "ICEWS05-15", "GDELT"]:
        print(f"\n==== Generating for {dataset} ====")
        generate_history_npz(dataset)
