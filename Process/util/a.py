"""

JoinABLe Joint Axis Prediction Network

"""


import argparse
import os
import sys
import json
import numpy as np
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from lightning_fabric.utilities.seed import seed_everything

# 假设这些是你项目中的本地模块
from utils import metrics
from torchmetrics.classification import JaccardIndex
from torchmetrics.classification import BinaryAccuracy
from utils import util
from datasets.joint_graph_dataset import JointGraphDataset
from models.joinable import JoinABLe


# 设置矩阵乘法精度以提高性能
torch.set_float32_matmul_precision('medium')  # 或 'high'


class JointPrediction(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.model = JoinABLe(
            self.hparams.hidden,
            self.hparams.input_features,
            dropout=self.hparams.dropout,
            mpn=self.hparams.mpn,
            batch_norm=self.hparams.batch_norm,
            reduction=self.hparams.reduction,
            post_net=self.hparams.post_net,
            pre_net=self.hparams.pre_net
        )



        self.test_iou = JaccardIndex(
            task="binary",
            num_classes=2,
            ignore_index=0
        )

        self.test_accuracy = BinaryAccuracy(
            threshold=self.hparams.threshold
        )

    def training_step(self, batch, batch_idx):
        g1, g2, jg = batch
        jg.edge_attr = jg.edge_attr.long()
        x = self.model(g1, g2, jg)
        # 注意：这里从 self.args 改为 self.hparams
        loss = self.model.compute_loss(self.hparams, x, jg)
        # Log the run at every epoch, although this gets reduced via mean to a float
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        g1, g2, jg = batch
        jg.edge_attr = jg.edge_attr.long()
        x = self.model(g1, g2, jg)
        loss = self.model.compute_loss(self.hparams, x, jg)
        top_1 = self.model.precision_at_top_k(x, jg.edge_attr, g1.num_nodes, g2.num_nodes, k=1)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=1, sync_dist=True)
        self.log("val_top_1", top_1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=1, sync_dist=True)
        return {"loss": loss, "top_1": top_1}


    def test_step(self, batch, batch_idx):
        # Get the split we are using from the dataset
        # split = self.test_dataloader.dataloader.dataset.split  # <-- 保留的注释
        split = self.hparams.test_split # 使用 hparams 中的参数
        # Inference
        g1, g2, jg = batch
        jg.edge_attr = jg.edge_attr.long()
        x = self.model(g1, g2, jg)
        loss = self.model.compute_loss(self.hparams, x, jg)
        # Get the probabilities and calculate metrics
        prob = F.softmax(x, dim=0) # 注意: 在多分类场景下，softmax 后需要选择对应正类的概率
        self.test_iou.update(prob, jg.edge_attr)
        self.test_accuracy.update(prob, jg.edge_attr)
        # Calculate the precision at k with a default sequence of k
        top_k = self.model.precision_at_top_k(x, jg.edge_attr, g1.num_nodes, g2.num_nodes)
        top_1 = top_k[0]
        self.log(f"eval_{split}_loss", loss, on_step=False, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        self.log(f"eval_{split}_top_1", top_1, on_step=False, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        # Log evaluation based on if there are holes or not
        # Batch size 1 and no shuffle lets us use the batch index

        # has_holes = self.test_dataloader.dataloader.dataset.has_holes[batch_idx] # <-- 保留的注释
        has_holes = None
        # has_holes = self.test_dataloader().dataset.has_holes[batch_idx] # <-- 保留的注释

        top_1_holes = None
        top_1_no_holes = None
        if has_holes:
            self.log(f"eval_{split}_top_1_holes", top_1, on_step=False, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
            top_1_holes = top_1
        else:
            self.log(f"eval_{split}_top_1_no_holes", top_1, on_step=False, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
            top_1_no_holes = top_1

        # 保存 step 输出到实例属性，供 on_test_epoch_end 使用
        if not hasattr(self, "_test_outputs"):
            self._test_outputs = []
        self._test_outputs.append({
            "top_k": top_k,
            "top_1_holes": top_1_holes,
            "top_1_no_holes": top_1_no_holes
        })

        return {
            "loss": loss,
            "top_k": top_k,
            "top_1_holes": top_1_holes,
            "top_1_no_holes": top_1_no_holes
        }

    def on_test_epoch_end(self):
        # Get the split we are using from the dataset
        # split = self.test_dataloader.dataloader.dataset.split # <-- 保留的注释
        split = self.hparams.test_split
        test_iou = self.test_iou.compute()
        test_accuracy = self.test_accuracy.compute()
        self.log(f"eval_{split}_iou", test_iou, sync_dist=True)
        self.log(f"eval_{split}_accuracy", test_accuracy, sync_dist=True)

        all_top_k = np.stack([x["top_k"] for x in self._test_outputs])
        all_top_1_holes = np.array([x["top_1_holes"] for x in self._test_outputs if x["top_1_holes"] is not None])
        all_top_1_no_holes = np.array([x["top_1_no_holes"] for x in self._test_outputs if x["top_1_no_holes"] is not None])
        # All samples should be either holes or no holes, so check the counts add up to the total
        assert len(all_top_1_holes) + len(all_top_1_no_holes) == all_top_k.shape[0]
        if len(all_top_1_holes) > 0:
            top_1_holes = all_top_1_holes.mean()
        else:
            top_1_holes = "--"
        if len(all_top_1_no_holes) > 0:
            top_1_no_holes = all_top_1_no_holes.mean()
        else:
            top_1_no_holes = "--"

        k_seq = metrics.get_k_sequence()
        top_k_values = metrics.calculate_precision_at_k_from_sequence(all_top_k, use_percent=False)
        top_k_results = ""
        for k, result in zip(k_seq, top_k_values):
            top_k_results += f"{k} {result * 100:.4f}%\n"
        self.print(f"Eval top-k results on {split} set:\n{top_k_results[:-2]}")
        
        # 确保 logger 总是列表
        loggers = self.logger if isinstance(self.logger, list) else [self.logger]

        for logger in loggers:
            # 如果是 CometLogger，才做 log_curve (此逻辑保留，尽管我们现在默认用 TensorBoardLogger)
            if "CometLogger" in str(type(logger)): # 更安全的检查方式
                logger.experiment.log_curve(
                    f"eval_{split}_top_k",
                    x=k_seq,
                    y=top_k_values.tolist(),
                    overwrite=True
                )

        self._test_outputs = [] # 清理缓存

        return {
            "iou": test_iou,
            "accuracy": test_accuracy,
            "top_1_holes": top_1_holes,
            "top_1_no_holes": top_1_no_holes
        }

    def forward(self, batch):
        # Used for inference
        g1, g2, jg = batch
        jg.edge_attr = jg.edge_attr.long()
        return self.model(g1, g2, jg)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, "min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def load_dataset(args, split="train", random_rotate=False, label_scheme="Joint", max_node_count=0,):
    return JointGraphDataset(
        root_dir=args.dataset,
        split=split,
        center_and_scale=True,
        random_rotate=random_rotate,
        delete_cache=args.delete_cache,
        limit=args.limit,
        threads=args.threads,
        label_scheme=label_scheme,
        max_node_count=max_node_count,
        input_features=args.input_features
    )


def main():
    # --- 1. 统一的参数解析 (合并自 args_common.py 和 args_train.py) ---
    parser = argparse.ArgumentParser("预测网络")

    # 执行流程
    parser.add_argument("traintest", type=str, default="traintest", choices=("train", "test", "traintest"), help="执行训练、测试或两者。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")

    # 路径与日志
    parser.add_argument("--dataset", type=str, default="data", help="数据集路径。")
    parser.add_argument("--exp_name", type=str, default="Prediction_Experiment", help="用于日志和文件命名的实验名称。")
    parser.add_argument("--checkpoint", type=str, default="last", help="评估时加载的检查点，例如 last, best。")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="用于恢复训练的完整检查点路径。")
    
    # 数据集参数
    parser.add_argument("--limit", type=int, default=0, help="限制数据样本数量。")
    parser.add_argument("--threads", type=int, default=8, help="数据加载时使用的线程数。")
    parser.add_argument("--num_workers", type=int, default=8, help="torch dataloader 使用的 worker 数量。")
    parser.add_argument("--delete_cache", action="store_true", default=False, help="删除数据集缓存。")
    parser.add_argument("--random_rotate", action="store_true", default=False, help="随机旋转训练集。")
    parser.add_argument("--train_label_scheme", type=str, default="Joint", help="训练时使用的标签方案。")
    parser.add_argument("--test_label_scheme", type=str, default="Joint,JointEquivalent", help="测试时使用的标签方案。")
    parser.add_argument("--test_split", type=str, default="test", choices=("train", "val", "test", "mix_test"), help="评估时使用的数据集分割。")
    parser.add_argument("--hole_scheme", type=str, default="both", choices=("holes", "no_holes", "both"), help="评估包含孔洞的几何体。")
    parser.add_argument("--max_node_count", type=int, default=950, help="限制图的最大节点数。")
    parser.add_argument("--max_nodes_per_batch", type=int, default=10000, help="动态批处理的最大节点数。")

    # 模型超参数
    parser.add_argument("--input_features", type=str, default="entity_types,length,face_reversed,edge_reversed", help="以逗号分隔的输入特征。")
    parser.add_argument("--hidden", type=int, default=384, help="隐藏单元数量。")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout 比率。")
    parser.add_argument("--mpn", type=str, choices=("gat", "gatv2"), default="gatv2", help="消息传递网络。")
    parser.add_argument("--pre_net", type=str, default="mlp", choices=("mlp", "cnn"), help="Pre-Net 类型。")
    parser.add_argument("--post_net", type=str, default="mlp", choices=("mm", "mlp"), help="Post-Net 方法。")
    parser.add_argument("--reduction", type=str, choices=("sum", "mean"), default="mean", help="损失规约方法。")
    parser.add_argument("--batch_norm", action="store_true", default=True, help="是否使用 BatchNorm。")

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数。")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小。")
    parser.add_argument("--lr", type=float, default=0.0001, help="初始学习率。")
    parser.add_argument("--loss", type=str, choices=("bce", "mle", "focal"), default="mle", help="损失函数。")
    parser.add_argument("--pos_weight", type=float, default=200.0, help="BCE 损失的正类权重。")
    parser.add_argument("--gamma", type=float, default=5.0, help="Focal loss 的 gamma 参数。")
    parser.add_argument("--alpha", type=float, default=0.25, help="Focal loss 的 alpha 参数。")
    parser.add_argument("--threshold", type=float, default=2.6E-06, help="准确率和 IoU 计算的阈值。")

    # 硬件与策略
    parser.add_argument("--accelerator", type=str, default="gpu", help="硬件加速器 ('gpu', 'cpu')。")
    parser.add_argument("--devices", nargs='+', type=int, default=[6], help="要使用的 GPU 设备列表 (例如: 0 1)。")
    parser.add_argument("--strategy", type=str, default="ddp", help="分布式训练策略 (例如: 'ddp')。")

    args = parser.parse_args()
    seed_everything(args.seed)


    # --- 2. 自动化的日志和路径管理 ---
    if args.resume_from_checkpoint:
        ckpt_path = Path(args.resume_from_checkpoint)
        version_dir = ckpt_path.parent.parent
        results_path = version_dir.parent.parent
        month_day = version_dir.parent.name
        run_version = version_dir.name
        print(f"从检查点恢复训练: {args.resume_from_checkpoint}")
        print(f"日志将继续写入: {version_dir}")
    else:
        results_path = Path("results") / args.exp_name
        month_day = time.strftime("%m%d")
        run_version = time.strftime("%H%M%S")

    logger = TensorBoardLogger(save_dir=str(results_path), name=month_day, version=run_version)
    log_dir = Path(logger.log_dir)

    # --- 3. 初始化模型和回调函数 ---
    model = JointPrediction(args)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename="best-{epoch:02d}", save_top_k=3, save_last=True, mode="min")
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last" # 保持与旧代码一致的 'last.ckpt' 命名

    # --- 4. 初始化 Trainer ---
    strategy = DDPStrategy(find_unused_parameters=False) if args.strategy == "ddp" and len(args.devices) > 1 else "auto"
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strategy,
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.epochs,
        sync_batchnorm=args.batch_norm if len(args.devices) > 1 else False,
        log_every_n_steps=50
    )

    # --- 5. 执行训练或测试 ---
    if args.traintest in ["train", "traintest"]:
        print(
            f"""
    -----------------------------------------------------------------------------------
    开始训练 模型
    -----------------------------------------------------------------------------------
    日志将写入: {log_dir}

    使用以下命令监控训练过程:
    tensorboard --logdir "{results_path}/{month_day}/{run_version}"

    最好的模型将保存在:
    {log_dir}/checkpoints/
    -----------------------------------------------------------------------------------
            """
        )
        train_dataset = load_dataset(args, split="train", random_rotate=args.random_rotate, label_scheme=args.train_label_scheme, max_node_count=args.max_node_count)
        val_dataset = load_dataset(args, split="val", label_scheme=args.test_label_scheme)
        train_loader = train_dataset.get_train_dataloader(max_nodes_per_batch=args.max_nodes_per_batch, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = val_dataset.get_test_dataloader(batch_size=1, num_workers=args.num_workers)
        
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)
        
        if trainer.global_rank == 0:
            print("--------------------------------------------------------------------------------")
            print("训练结果")
            for key, val in trainer.logged_metrics.items():
                print(f"{key}: {val}")
            print("--------------------------------------------------------------------------------")

    if args.traintest in ["test", "traintest"]:
        print("\n############################################ 开始评估模型... ##################################################")
        
        # 确定 checkpoint 路径
        if args.resume_from_checkpoint:
            ckpt_path = args.resume_from_checkpoint
        else:
            ckpt_path_obj = log_dir / "checkpoints" / f"{args.checkpoint}.ckpt"
            if not ckpt_path_obj.exists():
                 # 备用方案：如果 `best.ckpt` 存在而 `last.ckpt` 不存在，尝试 `best`
                 ckpt_path_obj = log_dir / "checkpoints" / "best.ckpt"
            ckpt_path = str(ckpt_path_obj)

        assert os.path.exists(ckpt_path), f"错误: 找不到检查点文件 at {ckpt_path}"
        
        print(f"正在评估检查点 {ckpt_path} 于 {args.test_split} 数据集")
        
        # 如果模型刚刚训练完，可以直接用于测试，否则从文件加载
        model_to_test = model if trainer.state.finished else JointPrediction.load_from_checkpoint(ckpt_path)
        
        test_dataset = load_dataset(args, split=args.test_split, label_scheme=args.test_label_scheme)
        test_loader = test_dataset.get_test_dataloader(batch_size=1, num_workers=args.num_workers)
        
        trainer.test(model_to_test, test_loader)

if __name__ == "__main__":
    # 确保在多进程环境中安全
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()