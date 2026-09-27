import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.optim import AdamW
import joblib
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix, AUROC
import os
import argparse
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')


@dataclass
class TrainingConfig:
    """二分类情感分析训练配置"""
    # 必要参数
    data_file:  str # 数据集CSV文件路径
    model_path: str  # BERT模型路径（本地或HuggingFace）
    output_dir: str  # 输出目录

    # 数据参数
    text_column: str = "review"  # 文本列名
    label_column: str = "label"  # 标签列名
    test_size: float = 0.2  # 测试集比例
    max_len: int = 128  # 最大文本长度

    # 训练参数
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # 模型参数
    dropout_rate: float = 0.3
    use_focal_loss: bool = False  # 是否使用Focal Loss处理不平衡
    focal_gamma: float = 2.0

    # 训练控制
    patience: int = 3  # 早停耐心值
    min_delta: float = 1e-4  # 改进阈值
    save_test_set: bool = True  # 是否保存测试集
    random_seed: int = 42
    gradient_accumulation_steps: int = 1

    # 设备配置
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    ))

    def __post_init__(self):
        """创建输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存配置
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            # 转换为可序列化的字典
            config_dict = self.__dict__.copy()
            # 处理field生成的值
            if isinstance(config_dict['device'], str):
                config_dict['device'] = config_dict['device']
            json.dump(config_dict, f, indent=2, ensure_ascii=False)


class BinaryTextDataset(Dataset):
    """二分类文本数据集"""

    def __init__(self, texts: np.ndarray, labels: np.ndarray,
                 tokenizer: BertTokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BinaryBERTClassifier(nn.Module):
    """二分类BERT模型"""

    def __init__(self, model_path: str, dropout_rate: float = 0.3):
        super().__init__()
        # 加载预训练BERT
        self.bert = BertModel.from_pretrained(model_path)#from_tf=True使用google模型tensflow需要先进行转换

        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

        # 初始化分类头
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        # BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 使用[CLS] token的表示
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # 分类
        logits = self.classifier(pooled_output)
        return logits


class BinaryFocalLoss(nn.Module):
    """二分类Focal Loss"""

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 计算交叉熵损失（每个样本）
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none'
        )

        # 获取预测概率
        probs = torch.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.view(-1, 1)).squeeze()

        # Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class BinaryMetrics:
    """二分类评估指标"""

    def __init__(self, device: str = 'cpu'):
        self.device = device

        # 初始化指标
        self.accuracy = Accuracy(task='binary').to(device)
        self.f1 = F1Score(task='binary').to(device)
        self.precision = Precision(task='binary').to(device)
        self.recall = Recall(task='binary').to(device)
        self.auroc = AUROC(task='binary').to(device)
        self.confusion = ConfusionMatrix(task='binary', num_classes=2).to(device)

        # 存储历史
        self.history = []

    def reset(self) -> None:
        """重置所有指标"""
        self.accuracy.reset()
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()
        self.auroc.reset()
        self.confusion.reset()

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """更新指标状态"""
        # 获取预测概率和类别
        probs = torch.softmax(logits, dim=1)[:, 1]  # 正类概率
        preds = torch.argmax(logits, dim=1)

        # 更新指标
        self.accuracy.update(preds, labels)
        self.f1.update(preds, labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)
        self.auroc.update(probs, labels)
        self.confusion.update(preds, labels)

    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        return {
            'accuracy': self.accuracy.compute().item(),
            'f1': self.f1.compute().item(),
            'precision': self.precision.compute().item(),
            'recall': self.recall.compute().item(),
            'auroc': self.auroc.compute().item(),
            'confusion_matrix': self.confusion.compute().cpu().numpy()
        }

    def compute_and_log(self, phase: str, epoch: int) -> Dict[str, float]:
        """计算指标并记录到历史"""
        metrics = self.compute()
        metrics['phase'] = phase
        metrics['epoch'] = epoch
        self.history.append(metrics)
        return metrics


class DataProcessor:
    """数据处理类"""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # 设置随机种子
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

    def load_data(self) -> pd.DataFrame:
        """加载CSV数据"""
        try:
            # 尝试不同编码读取
            encodings = ['utf-8', 'gbk', 'latin1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(self.config.data_file, encoding=encoding)
                    print(f"成功使用 {encoding} 编码读取数据")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("无法读取文件，请检查文件编码")

            # 检查必要的列
            required_cols = [self.config.text_column, self.config.label_column]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"缺少必要的列: {missing_cols}")

            # 数据清洗
            print(f"原始数据大小: {len(df)}")

            # 移除空值
            df = df.dropna(subset=[self.config.text_column, self.config.label_column])

            # 确保标签为0/1
            unique_labels = df[self.config.label_column].unique()
            if len(unique_labels) != 2:
                # 尝试将非0/1标签映射为0/1
                if set(unique_labels) <= {0, 1}:
                    pass  # 已经是0/1
                else:
                    # 假设负向为0，正向为1
                    df[self.config.label_column] = df[self.config.label_column].astype(str)
                    # 这里根据实际情况调整，比如包含"负面"、"消极"等词为0
                    df[self.config.label_column] = df[self.config.label_column].apply(
                        lambda x: 0 if '负' in str(x) or '0' in str(x) or '消极' in str(x) else 1
                    )
                    print(f"已将标签映射为0/1，分布:\n{df[self.config.label_column].value_counts()}")

            print(f"清洗后数据大小: {len(df)}")
            return df

        except Exception as e:
            print(f"数据加载失败: {e}")
            raise

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """划分训练验证集和测试集"""
        # 分层抽样划分
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=df[self.config.label_column]
        )

        print(f"训练验证集: {len(train_val_df)} 条")
        print(f"测试集: {len(test_df)} 条")
        print(f"类别分布 - 训练集: {train_val_df[self.config.label_column].value_counts().to_dict()}")
        print(f"类别分布 - 测试集: {test_df[self.config.label_column].value_counts().to_dict()}")

        return train_val_df, test_df

    def save_test_set(self, test_df: pd.DataFrame) -> str:
        """保存测试集到本地"""
        test_set_path = os.path.join(self.config.output_dir, "test_set.csv")
        test_df.to_csv(test_set_path, index=False, encoding='utf-8')
        print(f"测试集已保存到: {test_set_path}")
        return test_set_path

    def create_data_loaders(self, train_val_df: pd.DataFrame,
                            test_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        # 初始化tokenizer
        tokenizer = BertTokenizer.from_pretrained(self.config.model_path)

        # 划分训练集和验证集 (80%训练，20%验证)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.2,  # 从训练验证集中再分20%作为验证集
            random_state=self.config.random_seed,
            stratify=train_val_df[self.config.label_column]
        )

        # 创建数据集
        train_dataset = BinaryTextDataset(
            train_df[self.config.text_column].values,
            train_df[self.config.label_column].values,
            tokenizer,
            self.config.max_len
        )

        val_dataset = BinaryTextDataset(
            val_df[self.config.text_column].values,
            val_df[self.config.label_column].values,
            tokenizer,
            self.config.max_len
        )

        test_dataset = BinaryTextDataset(
            test_df[self.config.text_column].values,
            test_df[self.config.label_column].values,
            tokenizer,
            self.config.max_len
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        print(f"训练集批次: {len(train_loader)}, 验证集批次: {len(val_loader)}, 测试集批次: {len(test_loader)}")

        return train_loader, val_loader, test_loader


class ModelTrainer:
    """模型训练器"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.metrics = BinaryMetrics(device=config.device)

        # 训练状态
        self.best_metrics = {
            'val_f1': 0.0,
            'val_loss': float('inf'),
            'epoch': -1
        }

    def setup_model(self) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
        """初始化模型、优化器和损失函数"""
        # 初始化模型
        model = BinaryBERTClassifier(self.config.model_path, self.config.dropout_rate)
        model.to(self.device)

        # 计算类别权重（处理不平衡数据）
        class_weights = None
        if self.config.use_focal_loss:
            # 简单假设正负类比例（实际应从训练数据计算）
            class_weights = torch.tensor([1.0, 1.0], device=self.device)

        # 损失函数
        if self.config.use_focal_loss:
            criterion = BinaryFocalLoss(
                gamma=self.config.focal_gamma,
                alpha=class_weights
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 优化器
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        return model, optimizer, criterion

    def train_epoch(self, model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module) -> float:
        """训练一个epoch"""
        model.train()
        total_loss = 0
        total_steps = 0

        pbar = tqdm(loader, desc="训练", leave=False)
        for batch in pbar:
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            # 梯度累积
            loss = loss / self.config.gradient_accumulation_steps

            # 反向传播
            loss.backward()

            # 更新参数
            if (total_steps + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # 记录损失
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_steps += 1

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(loader)

    def evaluate(self, model: nn.Module, loader: DataLoader,
                 criterion: nn.Module, phase: str = '验证') -> Tuple[float, Dict[str, float]]:
        """评估模型"""
        model.eval()
        total_loss = 0

        # 重置指标
        self.metrics.reset()

        with torch.no_grad():
            pbar = tqdm(loader, desc=phase, leave=False)
            for batch in pbar:
                # 移动到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 前向传播
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                # 记录损失
                total_loss += loss.item()

                # 更新指标
                self.metrics.update(logits, labels)

                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(loader)
        metrics = self.metrics.compute()

        return avg_loss, metrics

    def should_save_model(self, val_f1: float, val_loss: float,
                          epoch: int) -> Tuple[bool, str]:
        """判断是否应该保存模型（最佳模型准则）"""
        save_model = False
        reason = ""

        # 准则1: F1分数提升超过阈值
        f1_improvement = val_f1 - self.best_metrics['val_f1']
        if f1_improvement > self.config.min_delta:
            save_model = True
            reason = f"F1分数提升: {f1_improvement:.4f}"

        # 准则2: F1分数相近但损失更低（处理F1分数波动）
        elif abs(f1_improvement) <= self.config.min_delta:
            loss_improvement = self.best_metrics['val_loss'] - val_loss
            if loss_improvement > self.config.min_delta:
                save_model = True
                reason = f"F1相近但损失降低: {loss_improvement:.4f}"

        # 更新最佳指标
        if save_model:
            self.best_metrics.update({
                'val_f1': val_f1,
                'val_loss': val_loss,
                'epoch': epoch
            })

        return save_model, reason

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              test_loader: DataLoader) -> Dict[str, Any]:
        """完整的训练流程"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)

        # 初始化模型
        model, optimizer, criterion = self.setup_model()

        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'test_metrics': None
        }

        # 早停相关
        no_improve_epochs = 0

        # 训练循环
        for epoch in range(self.config.epochs):
            print(f"\n{'=' * 40}")
            print(f"Epoch {epoch + 1}/{self.config.epochs}")
            print(f"{'=' * 40}")

            # 训练
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            history['train_loss'].append(train_loss)

            # 验证
            val_loss, val_metrics = self.evaluate(model, val_loader, criterion, '验证')
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)

            # 打印验证结果
            print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
            print(f"验证指标 - F1: {val_metrics['f1']:.4f}, "
                  f"准确率: {val_metrics['accuracy']:.4f}, "
                  f"AUC: {val_metrics['auroc']:.4f}")

            # 检查是否应该保存模型
            should_save, reason = self.should_save_model(
                val_metrics['f1'], val_loss, epoch
            )

            if should_save:
                # 保存最佳模型
                model_save_path = os.path.join(self.config.output_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_metrics['f1'],
                    'val_loss': val_loss,
                    'config': self.config
                }, model_save_path)

                # 保存混淆矩阵
                conf_matrix = val_metrics['confusion_matrix']
                np.save(os.path.join(self.config.output_dir, "confusion_matrix.npy"), conf_matrix)

                print(f"✓ 保存最佳模型: {reason}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                print(f"未改进，早停计数: {no_improve_epochs}/{self.config.patience}")

            # 检查早停
            if no_improve_epochs >= self.config.patience:
                print(f"早停触发于第{epoch + 1}轮")
                break

        # 加载最佳模型进行最终测试
        print("\n" + "=" * 60)
        print("使用最佳模型进行测试集评估")
        print("=" * 60)

        # 加载最佳模型
        checkpoint = torch.load(os.path.join(self.config.output_dir, "best_model.pt"),weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 测试集评估
        test_loss, test_metrics = self.evaluate(model, test_loader, criterion, '测试')
        history['test_metrics'] = test_metrics

        # 保存最终模型
        final_model_path = os.path.join(self.config.output_dir, "final_model.pt")
        torch.save({
            'epoch': checkpoint['epoch'],
            'model_state_dict': model.state_dict(),
            'test_metrics': test_metrics,
            'val_metrics': checkpoint['val_f1'],
            'config': self.config,
            'training_history': history
        }, final_model_path)

        # 保存训练历史
        history_path = os.path.join(self.config.output_dir, "training_history.json")

        # 深度转换所有numpy数组为列表
        def convert_numpy(obj):
            """递归转换numpy数组和标量为Python原生类型"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
                return obj.item()  # 转换为Python原生类型
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        # 转换整个history对象
        history_for_save = convert_numpy(history)

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_for_save, f, indent=2, ensure_ascii=False)

        # 打印最终结果
        print(f"\n最佳模型在验证集上的表现:")
        print(f"  F1分数: {checkpoint['val_f1']:.4f} (Epoch {checkpoint['epoch'] + 1})")

        print(f"\n测试集最终评估:")
        print(f"  测试损失: {test_loss:.4f}")
        print(f"  准确率: {test_metrics['accuracy']:.4f}")
        print(f"  F1分数: {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  AUC-ROC: {test_metrics['auroc']:.4f}")

        # 混淆矩阵
        conf_matrix = test_metrics['confusion_matrix']
        print(f"\n混淆矩阵:")
        print(f"         预测负类   预测正类")
        print(f"真实负类  {conf_matrix[0, 0]:>6}      {conf_matrix[0, 1]:>6}")
        print(f"真实正类  {conf_matrix[1, 0]:>6}      {conf_matrix[1, 1]:>6}")

        return history


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BERT二分类情感分析训练')

    # 必要参数
    parser.add_argument('--data_file', type=str, required=False,default='sentiment_data/weibo_sentiment.csv',  # 你的默认路径
                        help='CSV数据文件路径，包含label和text列')
    parser.add_argument('--model_path', type=str, required=False,default='bert_base_chinese',
                        help='BERT模型路径（本地或HuggingFace模型名称）')
    parser.add_argument('--output_dir', type=str, required=False,default='bert_output',
                        help='输出目录，用于保存模型和结果')

    # 可选参数
    parser.add_argument('--text_column', type=str, default='review',
                        help='文本列名（默认：text）')
    parser.add_argument('--label_column', type=str, default='label',
                        help='标签列名（默认：label）')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例（默认：0.2）')
    parser.add_argument('--max_len', type=int, default=128,
                        help='最大文本长度（默认：128）')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小（默认：16）')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数（默认：10）')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率（默认：2e-5）')
    parser.add_argument('--patience', type=int, default=3,
                        help='早停耐心值（默认：3）')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='使用Focal Loss处理类别不平衡')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子（默认：42）')#                 固定每次的训练集不变

    args = parser.parse_args()

    # 创建配置
    config = TrainingConfig(
        data_file=args.data_file,
        model_path=args.model_path,
        output_dir=args.output_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        test_size=args.test_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        use_focal_loss=args.use_focal_loss,
        random_seed=args.random_seed
    )

    try:
        # 数据准备
        processor = DataProcessor(config)

        # 加载数据
        print("正在加载数据...")
        df = processor.load_data()

        # 划分数据
        print("正在划分数据集...")
        train_val_df, test_df = processor.split_data(df)

        # 保存测试集
        if config.save_test_set:
            processor.save_test_set(test_df)

        # 创建数据加载器
        print("正在创建数据加载器...")
        train_loader, val_loader, test_loader = processor.create_data_loaders(
            train_val_df, test_df
        )

        # 训练模型
        trainer = ModelTrainer(config)
        history = trainer.train(train_loader, val_loader, test_loader)

        print(f"\n训练完成！所有文件保存在: {config.output_dir}")
        print("\n生成的文件:")
        print(f"  - 最佳模型: {os.path.join(config.output_dir, 'best_model.pt')}")
        print(f"  - 最终模型: {os.path.join(config.output_dir, 'final_model.pt')}")
        print(f"  - 测试集: {os.path.join(config.output_dir, 'test_set.csv')}")
        print(f"  - 训练历史: {os.path.join(config.output_dir, 'training_history.json')}")
        print(f"  - 配置: {os.path.join(config.output_dir, 'config.json')}")

    except Exception as e:
        print(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()