import os
import torch
import argparse
import torch.optim as optim
from tqdm import tqdm
import sys
import time
import numpy as np
import csv
import swanlab

from datasets import TemporalDataset,HistoryFreqFilter
from models import SFTeAST,SimfyScorer
from regularizers import N3, Lambda3, Linear3, Spiral3
from optimizers import TKBCOptimizer
from simfy_model import Simfy

def get_args():
    parser = argparse.ArgumentParser(description="Train TeAST Temporal KGE Model")

    # --- 基础参数 ---
    parser.add_argument("--dataset", type=str, default="GDELT", help="Dataset name")
    parser.add_argument("--model", type=str, default="SFTeAST", help="Model name") #
    parser.add_argument("--rank", type=int, default=2000, help="Embedding dimension / 2")

    # --- 训练参数 ---
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-1)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--valid_freq", type=int, default=5)
    parser.add_argument("--early_stop", type=int, default=10)

    # --- 正则化 ---
    parser.add_argument("--emb_reg", type=float, default=0.0025)
    parser.add_argument("--time_reg", type=float, default=0.01)
    parser.add_argument( "--reg_type", type=str, default="Spiral3",
        choices=["N3", "Lambda3", "Linear3", "Spiral3"], help="Regularization strategy for time embeddings")

    # --- 扩展模块 ---
    parser.add_argument("--no_time_emb", action="store_true", help="Disable temporal embedding")
    parser.add_argument("--use_freq_filter", action="store_true", help="Enable history frequency filter")
    parser.add_argument("--freq_threshold", type=float, default=0.8, help="Keep top (1 - threshold) candidates")
    parser.add_argument("--freq_topk", type=int, default=None, help="Keep top-k candidates")
    parser.add_argument("--use_simfy", action="store_true", help="Enable SimFy structural scorer")
    parser.add_argument("--gamma", type=float, default=0.1, help="Weight for SimFy score")
    parser.add_argument("--alpha", type=float, default=0.3, help="Initial value for TeAST weight parameter")

    # --- 目录 ---
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save model checkpoints")

    return parser.parse_args()

def avg_both(mrrs, hits):
    """聚合左右预测结果"""
    m = (mrrs['lhs'] + mrrs['rhs']) / 2
    h = (hits['lhs'] + hits['rhs']) / 2
    return {'MRR': m, 'hits@[1,3,10]': h}


class Trainer:
    def __init__(self, model, dataset, optimizer, emb_reg, time_reg, args):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.emb_reg = emb_reg
        self.time_reg = time_reg
        self.args = args

        # 保存路径
        # self.path = args.save_dir

        os.makedirs(self.args.save_dir, exist_ok=True)

        self.best_mrr = 0
        self.patience = 0


    def train_epoch(self, examples):
        """单轮训练"""
        self.model.train()
        optimizer = TKBCOptimizer(
            self.model, self.emb_reg, self.time_reg, self.optimizer,
            batch_size=self.args.batch_size
        )
        optimizer.epoch(examples)

    def evaluate(self):
        """在valid/test/train上评估"""
        results = {}
        for split in tqdm(['valid', 'test', 'train'], desc="Evaluating", leave=False):
            cutoff = -1 if split != 'train' else 50000
            mrrs, hits = self.dataset.eval(self.model, split, cutoff)
            results[split] = avg_both(mrrs, hits)


        valid, test, train = results['valid'], results['test'], results['train']
        print(f"[VALID] {valid['MRR']:.4f}  [TEST] {test['MRR']:.4f}  [TRAIN] {train['MRR']:.4f}")
        # swanlab.log({
        #     "Valid_MRR": valid['MRR'],
        #     "Test_MRR": test['MRR'],
        #     "Train_MRR": train['MRR'],
        #     "Valid_Hits@1": valid['hits@[1,3,10]'][0],
        #     "Valid_Hits@3": valid['hits@[1,3,10]'][1],
        #     "Valid_Hits@10": valid['hits@[1,3,10]'][2],
        # })
        return valid, test, train

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir,
                                                         f"{self.args.model}_{self.args.dataset}_rank{self.args.rank}_lr{self.args.learning_rate}_emb{self.args.emb_reg}_time{self.args.time_reg}_alpha{self.args.alpha}_threshold{self.args.freq_threshold}.pkl"))

    def fit(self):
        """完整训练过程"""
        examples = torch.from_numpy(self.dataset.get_train().astype('int64'))

        progress_bar = tqdm(range(self.args.max_epochs), desc="Training Epochs", ncols=100)
        epoch_times = []
        for epoch in progress_bar:
            start_time = time.time()
            self.train_epoch(examples)
            end_time = time.time()
            epoch_time = end_time - start_time
            epoch_times.append(epoch_time)
            # 记录每个 epoch 的 loss / reg
            # swanlab.log({
            #     "epoch": epoch + 1,
            #     "emb_reg": self.emb_reg.weight if hasattr(self.emb_reg, 'weight') else self.args.emb_reg,
            #     "time_reg": self.time_reg.weight if hasattr(self.time_reg, 'weight') else self.args.time_reg,
            #     "epoch_time": epoch_time,
            # })

            # 每隔 valid_freq 评估一次
            if (epoch + 1) % self.args.valid_freq == 0:
                valid, test, _ = self.evaluate()
                mrr = valid['MRR']
                progress_bar.set_postfix({"Valid MRR": f"{mrr:.4f}"})

                if mrr > self.best_mrr:
                    self.best_mrr = mrr
                    self.patience = 0
                    self.save_checkpoint()
                    tqdm.write(f"✅ Epoch {epoch+1}: New best MRR = {mrr:.4f}")

                else:
                    self.patience += 1
                    tqdm.write(f"⏸ Epoch {epoch+1}: No improvement ({self.patience}/{self.args.early_stop})")
                    if self.patience >= self.args.early_stop:
                        tqdm.write("⏹ Early stopping triggered.")
                        break

        # 最终测试
        self.model.load_state_dict(torch.load(os.path.join(self.args.save_dir,
                                                           f"{self.args.model}_{self.args.dataset}_rank{self.args.rank}_lr{self.args.learning_rate}_emb{self.args.emb_reg}_time{self.args.time_reg}_alpha{self.args.alpha}_threshold{self.args.freq_threshold}.pkl")))
        final = avg_both(*self.dataset.eval(self.model, 'test', -1))
        tqdm.write(f"\n🎯 Final TEST MRR = {final['MRR']:.4f} | hits@[1,3,10] = {final['hits@[1,3,10]']} ")
        # swanlab.log({
        #     "Final/Test_MRR": final["MRR"],
        #     "Final/Hits@1": final["hits@[1,3,10]"][0],
        #     "Final/Hits@3": final["hits@[1,3,10]"][1],
        #     "Final/Hits@10": final["hits@[1,3,10]"][2],
        # })


def main():
    args = get_args()
    # 初始化 SawLab（wandb）
    # swanlab.login("B8QVC3XpszQSj2iqmRUTG")
    # swanlab.init(
    #     project=args.model,
    #     experiment_name=f"{args.dataset}_rank{args.rank}_lr{args.learning_rate}",
    #     config=vars(args)
    # )

    dataset = TemporalDataset(args.dataset)
    sizes = dataset.get_shape()

    # 模型初始化
    model = SFTeAST(sizes, args.rank, args.gamma, args.alpha, no_time_emb=args.no_time_emb).cuda()

    # ---- 历史频率过滤 ----
    if args.use_freq_filter:
        model.freq_filter = HistoryFreqFilter(
            args.dataset, threshold=args.freq_threshold, topk=args.freq_topk
        )

    # ---- SimFy 结构评分模块 ----
    if args.use_simfy:
        simfy_path = f"TEST_RE/models/{args.dataset}/models/{args.dataset}_best.pth"
        sys.modules["simfy_model"] = sys.modules[__name__]
        simfy_model = torch.load(simfy_path, map_location="cpu")

        model.simfy_scorer = SimfyScorer(
            simfy_model.entity_embeds,
            simfy_model.rel_embeds,
            simfy_model.similarity_pred_layer,
        ).cuda()
        model.simfy_scorer.eval()
        # model.gamma_sim = args.gamma_sim
        print(f"[INFO] Loaded SimFy scorer (alpha={args.alpha})")

    # 优化器
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)

    # 正则化器选择
    reg_map = {
        "N3": N3(args.emb_reg),
        "Lambda3": Lambda3(args.time_reg),
        "Linear3": Linear3(args.time_reg),
        "Spiral3": Spiral3(args.time_reg)
    }
    emb_reg = reg_map["N3"]
    time_reg = reg_map[args.reg_type]

    # 训练器封装
    trainer = Trainer(model, dataset, optimizer, emb_reg, time_reg, args)
    trainer.fit()
    # swanlab.finish()


if __name__ == '__main__':
    main()
