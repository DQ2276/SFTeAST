import os
import sys
import json
import csv
import torch
import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_rel

from datasets import TemporalDataset, HistoryFreqFilter
from models import SFTeAST, SimfyScorer
from simfy_model import Simfy


# =========================================================
# 参数
# =========================================================
def get_args():
    parser = argparse.ArgumentParser(description="Unified Evaluation & Analysis for SFTeAST")

    # 基本参数
    parser.add_argument("--dataset", type=str, default="ICEWS14")
    parser.add_argument("--model", type=str, default="SFTeAST")
    parser.add_argument("--model_path", type=str, default="models")
    parser.add_argument("--rank", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--emb_reg", type=float, default=0.0025)
    parser.add_argument("--time_reg", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 可选模块
    parser.add_argument("--use_simfy", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.9, help="Initial value for TeAST weight parameter")
    parser.add_argument("--use_freq_filter", action="store_true")
    parser.add_argument("--freq_threshold", type=float, default=0.99)
    parser.add_argument("--freq_topk", type=int, default=None)

    # 控制执行项
    parser.add_argument("--save_ranks", action="store_true", help="保存 ranks.npy")
    parser.add_argument("--evaluate", action="store_true", help="验证实验")
    parser.add_argument("--pattern_test", action="store_true", help="执行模式验证实验")
    parser.add_argument("--t_test", action="store_true", help="执行显著性检验")
    parser.add_argument("--baseline", type=str, default="TeAST", help="t-test 基准模型名")

    # 模式验证参数
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--save_csv", type=str, default="results/pattern_results.csv")

    return parser.parse_args()


# =========================================================
# 标准评估（含 ranks 保存）
# =========================================================
@torch.no_grad()
def evaluate(model, dataset, save_dir=None, model_name="SFTeAST", save_ranks=False):
    model.eval()
    print("\n[INFO] Evaluating on test set ...")
    mrrs, hits, ranks = dataset.eval(model, "test", -1, result_ranks=True)
    mrr = (mrrs["lhs"] + mrrs["rhs"]) / 2
    hits_avg = (hits["lhs"] + hits["rhs"]) / 2

    results = {
        "MRR": float(mrr),
        "Hits@1": float(hits_avg[0]),
        "Hits@3": float(hits_avg[1]),
        "Hits@10": float(hits_avg[2]),
    }

    print("\n===== Evaluation Results =====")
    for k, v in results.items():
        print(f"{k:8s}: {v:.4f}")

    if save_ranks:
        np.save(save_dir, ranks.cpu().numpy())
        print(f"[INFO] Saved ranks to {save_dir}")

    return results

# =========================================================
# 模式验证
# =========================================================
@torch.no_grad()
def check_pattern_symmetry(model, num_e, num_r, num_t):
    """
    对称性模式：检查 (s, r, o, t) 与 (o, r, s, t) 是否得分一致
    """
    device = next(model.parameters()).device  # 自动检测模型所在设备
    s1, s2 = torch.randint(0, num_e, (2,), device=device)
    r = torch.randint(0, num_r, (1,), device=device)
    t = torch.randint(0, num_t, (1,), device=device)
    x1 = torch.tensor([[s1, r, s2, t]], device=device)
    x2 = torch.tensor([[s2, r, s1, t]], device=device)
    return torch.allclose(model.score(x1), model.score(x2), atol=1e-2)

@torch.no_grad()
def check_pattern_inverse(model, num_e, num_r, num_t):
    """
    逆关系模式：检查 (s, r1, o, t) 与 (o, r2, s, t) 是否得分一致
    """
    device = next(model.parameters()).device
    s, o = torch.randint(0, num_e, (2,), device=device)
    r1, r2 = torch.randint(0, num_r, (2,), device=device)
    t = torch.randint(0, num_t, (1,), device=device)
    x1 = torch.tensor([[s, r1, o, t]], device=device)
    x2 = torch.tensor([[o, r2, s, t]], device=device)
    return torch.allclose(model.score(x1), model.score(x2), atol=1e-2)

@torch.no_grad()
def check_pattern_temporal(model, num_e, num_r, num_t):
    """
    时间敏感模式：同一三元组在不同时间得分应不同
    """
    device = next(model.parameters()).device
    s, o = torch.randint(0, num_e, (1,), device=device), torch.randint(0, num_e, (1,), device=device)
    r = torch.randint(0, num_r, (1,), device=device)
    t1, t2 = torch.randint(0, num_t, (2,), device=device)
    x1 = torch.tensor([[s, r, o, t1]], device=device)
    x2 = torch.tensor([[s, r, o, t2]], device=device)
    return not torch.allclose(model.score(x1), model.score(x2), atol=1e-2)

def run_pattern_test(model, num_e, num_r, num_t):
    """
    单次验证三种模式（True/False）
    """
    sym_ok = check_pattern_symmetry(model, num_e, num_r, num_t)
    inv_ok = check_pattern_inverse(model, num_e, num_r, num_t)
    temp_ok = check_pattern_temporal(model, num_e, num_r, num_t)

    print("\n=== Pattern Verification Result ===")
    print(f"对称性 (Symmetry) : {'✔ Yes' if sym_ok else '✖ No'}")
    print(f"逆关系 (Inverse)  : {'✔ Yes' if inv_ok else '✖ No'}")
    print(f"时间敏感 (Temporal): {'✔ Yes' if temp_ok else '✖ No'}")

    return {
        "sym": sym_ok,
        "inv": inv_ok,
        "temp": temp_ok
    }

# =========================================================
# 显著性检验
# =========================================================
def compute_metrics(ranks):
    mrr = 1.0 / ranks
    return mrr, (ranks <= 1).astype(float), (ranks <= 3).astype(float), (ranks <= 10).astype(float)

def paired_ttest(path_base, path_new):
    ranks_base = np.load(path_base)
    ranks_new = np.load(path_new)
    assert len(ranks_base) == len(ranks_new), "样本数量必须一致"

    mrr1, h1_1, h3_1, h10_1 = compute_metrics(ranks_base)
    mrr2, h1_2, h3_2, h10_2 = compute_metrics(ranks_new)

    results = {}
    for name, x1, x2 in [
        ("MRR", mrr1, mrr2),
        ("Hits@1", h1_1, h1_2),
        ("Hits@3", h3_1, h3_2),
        ("Hits@10", h10_1, h10_2),
    ]:
        t, p = ttest_rel(x2, x1)
        results[name] = (t, p)
    return results

# =========================================================
# 主流程
# =========================================================
def main():
    args = get_args()

    print(f"[INFO] Loading dataset: {args.dataset}")
    dataset = TemporalDataset(args.dataset)
    sizes = dataset.get_shape()

    model_dir = os.path.join(
        args.model_path,
        f"rank{args.rank}_lr{args.learning_rate}_emb{args.emb_reg}_time{args.time_reg}",
    )
    baseline_dir = os.path.join(
        args.model_path,
        f"rank{args.rank}_lr{args.learning_rate}_emb{args.emb_reg}_time{args.time_reg}",
    )
    save_file = f"{args.model}_{args.dataset}_rank{args.rank}_lr{args.learning_rate}_emb{args.emb_reg}_time{args.time_reg}_alpha{args.alpha}_threshold{args.freq_threshold}"
    # save_file = f"{args.model}_{args.dataset}_rank{args.rank}_lr{args.learning_rate}_emb{args.emb_reg}_time{args.time_reg}_threshold{args.freq_threshold}"

    model_file = os.path.join(args.model_path, f"{save_file}.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"❌ Model checkpoint not found: {model_file}")

    model = SFTeAST(sizes, args.rank, no_time_emb=False).to(args.device)
    if args.use_freq_filter:
        model.freq_filter = HistoryFreqFilter(args.dataset, threshold=args.freq_threshold, topk=args.freq_topk)
    if args.use_simfy:
        simfy_path = f"TEST_RE/models/{args.dataset}/models/{args.dataset}_best.pth"
        simfy_model = torch.load(simfy_path, map_location="cpu")
        model.simfy_scorer = SimfyScorer(
            simfy_model.entity_embeds,
            simfy_model.rel_embeds,
            simfy_model.similarity_pred_layer,
        ).to(args.device)
        model.simfy_scorer.eval()

    state_dict = torch.load(model_file, map_location=args.device)
    model.load_state_dict(state_dict)
    print("[INFO] Model loaded successfully!")

    # ---- 标准评估 ----
    if args.evaluate:
        results = evaluate(model, dataset, save_dir=os.path.join(args.model_path, f"{save_file}_test_ranks.npy"), model_name=args.model, save_ranks=args.save_ranks)
        print("results: " + str(results))
        # ---- 保存最终评估结果 ----
        # save_path = os.path.join(args.model_path, f"{save_file}.json")
        # with open(save_path, "w") as f:
        #     json.dump(results, f, indent=4)
        # print(f"\n[INFO] Results saved to {save_path}")

    # ---- 模式验证 ----
    if args.pattern_test:
        num_e, num_r, num_t = sizes[0], sizes[1], sizes[3]
        pattern_res = run_pattern_test(model, num_e, num_r, num_t)
        # with open(args.save_csv, "w", newline="") as f:
        #     writer = csv.DictWriter(f, fieldnames=["Model", "sym", "inv", "temp"])
        #     writer.writeheader()
        #     writer.writerow({"Model": args.model, **pattern_res})
        # print("\n=== Pattern Verification Summary ===")
        # print(f"{'Model':10s} {'Symmetry':>10s} {'Inverse':>10s} {'Temporal':>10s}")
        # print(f"{args.model:10s} {pattern_res['sym']*100:9.1f}% {pattern_res['inv']*100:9.1f}% {pattern_res['temp']*100:9.1f}%")

    # ---- 显著性检验 ----
    if args.t_test:
        path_base = os.path.join(args.model_path, f"{args.baseline}_{args.dataset}_rank{args.rank}_lr{args.learning_rate}_emb{args.emb_reg}_time{args.time_reg}_test_ranks.npy")
        path_new = os.path.join(args.model_path, f"{save_file}_test_ranks.npy")
        if not os.path.exists(path_base):
            raise FileNotFoundError(f"Baseline rank file not found: {path_base}")
        if not os.path.exists(path_new):
            raise FileNotFoundError(f"Improved rank file not found: {path_new}")
        res = paired_ttest(path_base, path_new)
        print(f"\n=== Paired t-test Results ({args.model} vs {args.baseline}) on {args.dataset} ===")
        for k, (t, p) in res.items():
            sig = "✅" if p < 0.05 else "❌"
            print(f"{k:8s}: t = {t:7.3f}, p = {p:9.6f} {sig}")




if __name__ == "__main__":
    main()

