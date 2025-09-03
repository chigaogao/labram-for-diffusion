import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import re
import csv
import argparse
import datetime
from collections import OrderedDict
import utils
# --- LaBraM 和相关库的导入 ---
import timm
from timm.models import create_model
import modeling_finetune
from utils import load_state_dict
from einops import rearrange
# --- 导入 LaBraM 的优化器工厂 (核心修改) ---
from optim_factory import create_optimizer, LayerDecayValueAssigner

# --- 从原脚本导入 ---
from eegdatasets_leaveone import EEGDataset
from loss import ClipLoss
from util import wandb_logger

os.environ["WANDB_MODE"] = 'offline'

class LaBraMFeatureExtractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.labram_backbone = create_model(
            args.model_name,
            pretrained=False,
            num_classes=1000,
            drop_path_rate=0.1,
            use_mean_pooling=True,
            init_scale=0.001,
            init_values=0.1
        )

        if args.finetune_path:
            # (加载权重的逻辑保持不变)
            print(f"Loading pre-trained weights from: {args.finetune_path}")
            checkpoint = torch.load(args.finetune_path, map_location='cpu')
            checkpoint_model = checkpoint.get('model', checkpoint)

            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                elif not key.startswith('head.'):
                    new_dict[key] = checkpoint_model[key]
            checkpoint_model = new_dict

            load_state_dict(self.labram_backbone, checkpoint_model, prefix='')
            print("Pre-trained weights loaded successfully.")

        feature_dim = self.labram_backbone.head.in_features
        self.labram_backbone.head = nn.Identity()

        self.proj_eeg = nn.Sequential(
            nn.Linear(feature_dim, args.proj_dim),
            nn.GELU(),
            nn.Linear(args.proj_dim, args.proj_dim),
            nn.LayerNorm(args.proj_dim)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x,ch_names):
        x = rearrange(x, 'B N (A T) -> B N A T', T=200)
        eeg_features = self.labram_backbone(x,input_chans=ch_names)
        out = self.proj_eeg(eeg_features)
        return out

    # --- (核心修改) 添加辅助方法以供 optim_factory 使用 ---
    def get_num_layers(self):
        """获取骨干网络中的Transformer层数"""
        return self.labram_backbone.get_num_layers()

    def no_weight_decay(self):
        """获取骨干网络中不需要权重衰减的参数名称集合"""
        return self.labram_backbone.no_weight_decay()

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    return int(match.group()) if match else None
def train_model(sub, eeg_model, dataloader, ch_names,optimizer, device, text_features_all, img_features_all, config):
    eeg_model.train()
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.90
    mse_loss_fn = nn.MSELoss()

    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)#64x63x200
        img_features = img_features.to(device).float()#64x1024
        labels = labels.to(device)

        optimizer.zero_grad()

        eeg_features = eeg_model(eeg_data,ch_names).float()

        logit_scale = eeg_model.logit_scale
        img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
        regress_loss = mse_loss_fn(eeg_features, img_features)
        loss = (alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        logits_img = logit_scale * eeg_features @ img_features_all.T
        predicted = torch.argmax(logits_img, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del eeg_data, eeg_features, img_features, text_features

    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    return average_loss, accuracy
def evaluate_model(sub, eeg_model, dataloader,ch_names, device, text_features_all, img_features_all, k, config):
    eeg_model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    correct = 0
    total = 0
    top5_correct_count = 0
    all_labels = set(range(text_features_all.size(0)))

    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            labels = labels.to(device)
            img_features = img_features.to(device).float()

            eeg_features = eeg_model(eeg_data,ch_names)

            logit_scale = eeg_model.logit_scale

            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k - 1) + [label.item()]
                random.shuffle(selected_classes)

                selected_img_features = img_features_all[selected_classes]

                logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T

                predicted_idx = torch.argmax(logits_img).item()
                predicted_label = selected_classes[predicted_idx]

                if predicted_label == label.item():
                    correct += 1

                if k >= 5:
                    _, top5_indices = torch.topk(logits_img, 5, largest=True)
                    top5_predicted_labels = [selected_classes[i] for i in top5_indices]
                    if label.item() in top5_predicted_labels:
                        top5_correct_count += 1

                total += 1
            del eeg_data, eeg_features, img_features

    accuracy = correct / total if total > 0 else 0
    top5_acc = top5_correct_count / total if total > 0 else 0
    return 0, accuracy, top5_acc
def main_train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader,ch_names, optimizer, device,
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all,
                    config):
    # 此函数完全不变
    logger = wandb_logger(config) if config.logger else None
    if logger:
        logger.watch(eeg_model, logger)

    best_accuracy = 0.0
    results = []

    for epoch in range(config.epochs):
        train_loss, train_accuracy = train_model(sub, eeg_model, train_dataloader,ch_names, optimizer, device,
                                                 text_features_train_all, img_features_train_all, config=config)

        if (epoch + 1) % 5 == 0:
            save_dir = f"./models/contrast/{config.encoder_type}/{sub}/{current_time}"
            os.makedirs(save_dir, exist_ok=True)
            file_path = f"{save_dir}/{epoch + 1}.pth"
            torch.save(eeg_model.state_dict(), file_path)
            print(f"Model saved to {file_path}")

        _, test_accuracy, top5_acc = evaluate_model(sub, eeg_model, test_dataloader,ch_names, device, text_features_test_all,
                                                    img_features_test_all, k=200, config=config)
        _, v2_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, ch_names,device, text_features_test_all,
                                      img_features_test_all, k=2, config=config)
        _, v4_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, ch_names,device, text_features_test_all,
                                      img_features_test_all, k=4, config=config)
        _, v10_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, ch_names,device, text_features_test_all,
                                       img_features_test_all, k=10, config=config)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

        print(
            f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {train_loss:.4f} | Test Acc@1: {test_accuracy:.4f} | Test Acc@5: {top5_acc:.4f} | V2 Acc: {v2_acc:.4f}")

        epoch_results = {
            "epoch": epoch + 1, "test_accuracy": test_accuracy, "top5_acc": top5_acc,
            "v2_acc": v2_acc, "v4_acc": v4_acc, "v10_acc": v10_acc
        }
        results.append(epoch_results)

        if logger:
            logger.log({
                "Train Loss": train_loss, "Train Accuracy": train_accuracy,
                "Test Accuracy": test_accuracy, "v2 Accuracy": v2_acc,
                "v4 Accuracy": v4_acc, "v10 Accuracy": v10_acc, "Epoch": epoch
            })

    if logger:
        logger.finish()
    return results
def main():
    parser = argparse.ArgumentParser(description='LaBraM Fine-tuning on Things-EEG with Advanced Strategy')

    # --- 数据和路径参数 ---
    parser.add_argument('--data_path', type=str, default="/root/autodl-fs/Preprocessed_data_200Hz",
                        help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast2', help='Directory to save output results')
    parser.add_argument('--subjects', nargs='+', default=['sub-01'], help='List of subject IDs')

    # --- 训练参数 ---
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    # --- 模型参数 ---
    parser.add_argument('--model_name', type=str, default='labram_base_patch200_200',
                        help='Name of the LaBraM model variant')
    parser.add_argument('--finetune_path', type=str,default="/root/autodl-fs/labram-base.pth",
                        help='Path to the pre-trained LaBraM model checkpoint (.pth file)')
    parser.add_argument('--proj_dim', type=int, default=1024, help='Dimension of the projection head output')


    parser.add_argument('--opt', default='adamw', type=str, help='Optimizer (default: "adamw")')
    parser.add_argument('--opt_eps', default=1e-8, type=float, help='Optimizer Epsilon')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', help='Optimizer Betas')
    parser.add_argument('--clip_grad', type=float, default=None, help='Clip gradient norm')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='Layer-wise learning rate decay from BEiT')


    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device to use')
    parser.add_argument('--logger', action='store_true', help='Enable WandB logging')
    parser.add_argument('--project', type=str, default="labram_things_eeg", help='WandB project name')
    parser.add_argument('--entity', type=str, default="your_entity", help='WandB entity name')
    parser.add_argument('--name', type=str, default="labram_finetune_advanced", help='Experiment name')

    args = parser.parse_args()
    args.encoder_type = 'LaBraM_Adv'
    args.insubject = True

    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    for sub in args.subjects:
        eeg_model = LaBraMFeatureExtractor(args)
        eeg_model.to(device)

        # --- (核心修改) 使用 LaBraM 的优化器创建逻辑 ---
        num_layers = eeg_model.get_num_layers()
        if args.layer_decay < 1.0:
            # 创建分层学习率分配器
            assigner = LayerDecayValueAssigner(
                list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
            )
        else:
            assigner = None

        # 获取不需要权重衰减的参数列表
        skip_weight_decay_list = eeg_model.no_weight_decay()
        # 将我们新添加的投影头和logit_scale也加入到优化器管理中
        # 默认情况下，新添加的层不使用层衰减，而是使用基础学习率
        # create_optimizer 能够自动处理这种情况
        optimizer = create_optimizer(
            args, eeg_model,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
        )

        # 数据加载部分保持不变
        train_dataset = EEGDataset(args.data_path, subjects=[sub], train=True)
        test_dataset = EEGDataset(args.data_path, subjects=[sub], train=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

        text_features_train_all = train_dataset.text_features#1654x1024
        text_features_test_all = test_dataset.text_features#200x1024
        img_features_train_all = train_dataset.img_features#16540X1024
        img_features_test_all = test_dataset.img_features#200X1024
        ch_names = ['FP1', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3',
				  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
				  'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
				  'CZ', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
				  'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
				  'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
				  'O1', 'OZ', 'O2']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        input_chans = utils.get_input_chans(ch_names)
        ch_names = input_chans
        results = main_train_loop(sub, current_time, eeg_model, train_loader, test_loader,ch_names, optimizer, device,
                                  text_features_train_all, text_features_test_all, img_features_train_all,
                                  img_features_test_all, config=args)

        # 保存结果部分保持不变
        results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
        os.makedirs(results_dir, exist_ok=True)
        results_file = f"{results_dir}/{args.encoder_type}_{sub}.csv"
        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            print(f'Results saved to {results_file}')


if __name__ == '__main__':
    main()