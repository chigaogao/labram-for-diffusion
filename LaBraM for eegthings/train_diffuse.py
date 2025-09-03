
from timm.models import create_model
import modeling_finetune # 导入此文件以注册LaBraM模型到timm
from loss import ClipLoss
from eegdatasets_leaveone import EEGDataset # 确保此文件存在
from argparse import Namespace
import os
import utils
train = False
classes = None
pictures = None
def load_data():
    data_list = []
    label_list = []
    texts = []
    images = []

    if train:
        text_directory = "/root/autodl-fs/images_set/training_images"
    else:
        text_directory = "/root/autodl-fs/images_set/test_images"

    dirnames = [d for d in os.listdir(text_directory) if os.path.isdir(os.path.join(text_directory, d))]
    dirnames.sort()

    if classes is not None:
        dirnames = [dirnames[i] for i in classes]

    for dir in dirnames:

        try:
            idx = dir.index('_')
            description = dir[idx + 1:]
        except ValueError:
            print(f"Skipped: {dir} due to no '_' found.")
            continue

        new_description = f"{description}"
        texts.append(new_description)

    if train:
        img_directory = "/root/autodl-fs/images_set/training_images"
    else:
        img_directory =  "/root/autodl-fs/images_set/test_images"

    all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
    all_folders.sort()

    if classes is not None and pictures is not None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            pic_idx = pictures[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                if pic_idx < len(all_images):
                    images.append(os.path.join(folder_path, all_images[pic_idx]))
    elif classes is not None and pictures is None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
    elif classes is None:
        images = []
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            images.extend(os.path.join(folder_path, img) for img in all_images)
    else:

        print("Error")
    return texts, images


texts, images = load_data()
# images
import os

import torch

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'

import random
from util import wandb_logger
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
import csv
from torch import Tensor
import itertools
import math
import re
import numpy as np
from loss import ClipLoss
import argparse
from torch import nn
from einops import rearrange
from torch.optim import AdamW
from torch.utils.data import DataLoader
class LaBraMFeatureExtractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.labram_backbone = create_model(
            args.model_name,
            pretrained=False, num_classes=1000, drop_path_rate=0.1,
            use_mean_pooling=True, init_scale=0.001, init_values=0.1
        )
        # exchange classification head
        feature_dim = self.labram_backbone.head.in_features
        self.labram_backbone.head = nn.Identity()
        # finetune head
        self.proj_eeg = nn.Sequential(
            nn.Linear(feature_dim, args.proj_dim), nn.GELU(),
            nn.Linear(args.proj_dim, args.proj_dim), nn.LayerNorm(args.proj_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

        # load model
        if args.finetune_path and os.path.exists(args.finetune_path):
            print(f"Loading full fine-tuned model state from: {args.finetune_path}")
            full_model_state_dict = torch.load(args.finetune_path, map_location='cpu')
            if 'model' in full_model_state_dict:
                full_model_state_dict = full_model_state_dict['model']
            self.load_state_dict(full_model_state_dict)
            print("Full fine-tuned model state loaded successfully.")
        else:
            raise FileNotFoundError(f"Weight file not found at {args.finetune_path}")

    def forward(self, x, ch_names):
        x = rearrange(x, 'B N (A T) -> B N A T', T=200)
        eeg_features = self.labram_backbone(x, input_chans=ch_names)
        out = self.proj_eeg(eeg_features)
        return out


def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None


def get_eegfeatures(sub, eegmodel, dataloader, device, text_features_all, img_features_all, k,Train=True,ch_names_indices=None):
    eegmodel.eval()
    text_features_all = text_features_all.to(device).float()#200x1024
    img_features_all = img_features_all.to(device).float()#200x1024
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.9
    top5_correct = 0
    top5_correct_count = 0

    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    mse_loss_fn = nn.MSELoss()
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    ridge_lambda = 0.1
    save_features = True
    features_list = []  # List to store features
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()

            batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
            subject_id = extract_id_from_string(sub)
            # eeg_data = eeg_data.permute(0, 2, 1)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            # if not config.insubject:
            #     subject_ids = torch.full((batch_size,), -1, dtype=torch.long).to(device)
            eeg_features = eeg_model(eeg_data, ch_names=ch_names_indices)
            features_list.append(eeg_features.cpu())
            regress_loss = mse_loss_fn(eeg_features, img_features)
            # print("eeg_features", eeg_features.shape)
            # print(torch.std(eeg_features, dim=-1))
            # print(torch.std(img_features, dim=-1))
            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            # loss = (regress_loss + ridge_lambda * l2_norm)
            img_loss = eegmodel.loss_func(eeg_features, img_features, logit_scale)
            text_loss = eegmodel.loss_func(eeg_features, text_features, logit_scale)
            contrastive_loss = img_loss
            regress_loss = mse_loss_fn(eeg_features, img_features)
            # print("text_loss", text_loss)
            # print("img_loss", img_loss)
            # print("regress_loss", regress_loss)
            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            # loss = (regress_loss + ridge_lambda * l2_norm)
            loss = alpha * regress_loss * 10 + (1 - alpha) * contrastive_loss * 10
            # print("loss", loss)
            total_loss += loss.item()

            for idx, label in enumerate(labels):

                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k - 1) + [label.item()]#打乱顺序后 ，把真正的标签放在最后一个位置
                selected_img_features = img_features_all[selected_classes]

                logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                # logits_single = (logits_text + logits_img) / 2.0
                logits_single = logits_img
                # print("logits_single", logits_single.shape)

                # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                predicted_label = selected_classes[
                    torch.argmax(logits_single).item()]  # (n_batch, ) \in {0, 1, ..., n_cls-1}
                if predicted_label == label.item():
                    correct += 1
                total += 1

        if save_features and Train==True:
            features_tensor = torch.cat(features_list, dim=0)
            print("features_tensor", features_tensor.shape)
            torch.save(features_tensor.cpu(), f"labram_eeg_features_{sub}_train.pt")  # Save features as .pt file
        else:
            features_tensor = torch.cat(features_list, dim=0)
            torch.save(features_tensor.cpu(), f"labram_eeg_features_{sub}_test.pt")  # Save features as .pt file
    # print("features_tensor", features_tensor.shape)
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    return average_loss, accuracy, labels, features_tensor.cpu()


config = {
    "data_path": "/root/autodl-fs/Preprocessed_data_200Hz",
    "epochs": 50,
    "batch_size": 1024,
    "logger": True,
    "encoder_type": 'labram',
}

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = config['data_path']
emb_img_test = torch.load('/root/autodl-tmp/Generation/variables/ViT-H-14_features_test.pt')
emb_img_train = torch.load('/root/autodl-tmp/Generation/variables/ViT-H-14_features_train.pt')
model_args = Namespace(
    model_name='labram_base_patch200_200',
    proj_dim=1024,
    finetune_path="/root/autodl-tmp/LaBraM for eegthings/models/contrast/LaBraM_Adv/sub-01/09-02_18-16/40.pth"
)
eeg_model =LaBraMFeatureExtractor(model_args)
eeg_model = eeg_model.to(device)
sub = 'sub-01'
#####################################################################################
test_dataset = EEGDataset(data_path, subjects=[sub], train=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)#顺序取样本
text_features_test_all = test_dataset.text_features
img_features_test_all = test_dataset.img_features
test_loss, test_accuracy, labels, eeg_features_test = get_eegfeatures(sub, eeg_model, test_loader, device,
                                                                      text_features_test_all, img_features_test_all,
                                                                     k=200,Train=False,ch_names_indices=input_chans)
print(f" - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")#14.5%
#####################################################################################
train_dataset = EEGDataset(data_path, subjects= [sub], train=True)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
text_features_test_all = train_dataset.text_features
img_features_test_all = train_dataset.img_features

train_loss, train_accuracy, labels, eeg_features_train = get_eegfeatures(sub, eeg_model, train_loader, device,
                                                                         text_features_test_all, img_features_test_all,k=200,Train=True,ch_names_indices=input_chans)
from diffusion_prior import *
from custom_pipeline import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
emb_img_train_4 = emb_img_train.view(1654,10,1,1024).repeat(1,1,4,1).view(-1,1024)
emb_eeg_train = torch.load('/root/autodl-tmp/LaBraM for eegthings/labram_eeg_features_sub-01_train.pt')
emb_eeg_test = torch.load('/root/autodl-tmp/LaBraM for eegthings/labram_eeg_features_sub-01_test.pt')
dataset = EmbeddingDataset(
    c_embeddings=emb_eeg_train, h_embeddings=emb_img_train_4,
    # h_embeds_uncond=h_embeds_imgnet
)
dl = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=64)#后面输入就变成1024x1024 1024x1024
diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
# number of parameters
print(sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))#9675648
pipe = Pipe(diffusion_prior, device=device)

# load pretrained model
model_name = 'diffusion_prior_labrambased' # 'diffusion_prior_vice_pre_imagenet' or 'diffusion_prior_vice_pre'
pipe.train(dl, num_epochs=150, learning_rate=1e-3) # to 0.142
# pipe.diffusion_prior.load_state_dict(torch.load(f'./fintune_ckpts/{config['encoder_type']}/{sub}/{model_name}.pt', map_location=device))

# pipe.diffusion_prior.load_state_dict(torch.load(f'./fintune_ckpts/{config['data_path']}/{sub}/{model_name}.pt', map_location=device))
save_path = f'./fintune_ckpts/{config["encoder_type"]}/{sub}/{model_name}.pt'

directory = os.path.dirname(save_path)

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)
torch.save(pipe.diffusion_prior.state_dict(), save_path)
