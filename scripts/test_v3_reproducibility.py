import io
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import zipfile
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing.image_preprocessing import preprocess_image
from preprocessing.text_preprocessing import Vocabulary, clean_caption

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)

class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_planes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(in_planes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class V3ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256, dropout_p=0.1):
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x, normalize=True):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

class V3TextEncoder(nn.Module):
    def __init__(self, vocab_size=158, word_embed_dim=128, hidden_dim=128, embed_dim=256, padding_idx=0, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(p=dropout_p)
        self.lstm = nn.LSTM(
            input_size=word_embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim * 2, embed_dim),
        )

    def forward(self, token_ids, normalize=True):
        mask = (token_ids != 0).unsqueeze(-1).float()
        emb = self.embed_dropout(self.embedding(token_ids))
        lstm_out, _ = self.lstm(emb)
        sum_masked = (lstm_out * mask).sum(dim=1)
        lens = mask.sum(dim=1).clamp(min=1e-9)
        pooled = sum_masked / lens
        proj = self.projection(pooled)
        if normalize:
            proj = F.normalize(proj, p=2, dim=-1)
        return proj

class V3DualEncoder(nn.Module):
    def __init__(self, embed_dim=256, vocab_size=158):
        super().__init__()
        self.image_encoder = V3ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = V3TextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.65926)

    def encode_image(self, images, normalize=True):
        return self.image_encoder(images, normalize=normalize)

    def encode_text(self, captions, normalize=True):
        return self.text_encoder(captions, normalize=normalize)

def main():
    with zipfile.ZipFile('models/WildMatch_AI_Member3_V3.zip') as z:
        dual_sd = torch.load(io.BytesIO(z.read('wildmatch_v3_dual_encoder.pth')), map_location='cpu')
        heads_sd = torch.load(io.BytesIO(z.read('wildmatch_v3_species_heads.pth')), map_location='cpu')
        v3_metrics = json.loads(z.read('wildmatch_v3_metrics.json').decode('utf-8'))

    dual_model = V3DualEncoder()
    dual_model.load_state_dict(dual_sd['model_state_dict'])
    dual_model.eval()

    img_head = nn.Linear(256, 4)
    img_head.load_state_dict(heads_sd['image_classifier_state_dict'])
    img_head.eval()

    txt_head = nn.Linear(256, 4)
    txt_head.load_state_dict(heads_sd['text_classifier_state_dict'])
    txt_head.eval()

    species_list = dual_sd['species']
    species2idx = {s: i for i, s in enumerate(species_list)}
    print('Species order:', species_list)

    test_df = pd.read_csv('dataset/splits/test.csv')
    vocab = Vocabulary.load_json('dataset/vocabulary.json')

    for use_imagenet_norm in [True, False]:
        img_tensors = []
        txt_tensors = []
        species_targets = []
        for _, row in test_df.iterrows():
            t = preprocess_image(row['image_path'])
            if use_imagenet_norm:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                t = (t - mean) / std
            img_tensors.append(t)
            
            caption_text = row['final_caption']
            cleaned = clean_caption(caption_text)
            token_ids = vocab.encode(cleaned, add_special_tokens=True, max_length=32)
            txt_tensors.append(torch.tensor(token_ids, dtype=torch.long))
            species_targets.append(species2idx[row['species']])

        imgs = torch.stack(img_tensors)
        txts = torch.stack(txt_tensors)
        targets = torch.tensor(species_targets, dtype=torch.long)

        with torch.no_grad():
            img_emb = dual_model.encode_image(imgs)
            txt_emb = dual_model.encode_text(txts)
            img_preds = img_head(img_emb).argmax(dim=-1)
            txt_preds = txt_head(txt_emb).argmax(dim=-1)
            sim = img_emb @ txt_emb.T

        img_acc = (img_preds == targets).float().mean().item()
        txt_acc = (txt_preds == targets).float().mean().item()

        # Image-to-Text retrieval
        i2t_exact_r1 = (sim.argmax(dim=1) == torch.arange(len(test_df))).float().mean().item()
        top5_i2t = sim.topk(5, dim=1).indices
        i2t_exact_r5 = (top5_i2t == torch.arange(len(test_df)).unsqueeze(1)).any(dim=1).float().mean().item()

        i2t_species_r1 = (targets[sim.argmax(dim=1)] == targets).float().mean().item()
        i2t_species_r5 = (targets[top5_i2t] == targets.unsqueeze(1)).any(dim=1).float().mean().item()

        # Text-to-Image retrieval
        t2i_sim = sim.T
        t2i_exact_r1 = (t2i_sim.argmax(dim=1) == torch.arange(len(test_df))).float().mean().item()
        top5_t2i = t2i_sim.topk(5, dim=1).indices
        t2i_exact_r5 = (top5_t2i == torch.arange(len(test_df)).unsqueeze(1)).any(dim=1).float().mean().item()

        t2i_species_r1 = (targets[t2i_sim.argmax(dim=1)] == targets).float().mean().item()
        t2i_species_r5 = (targets[top5_t2i] == targets.unsqueeze(1)).any(dim=1).float().mean().item()

        print(f"\n=== Results (use_imagenet_norm={use_imagenet_norm}) ===")
        print(f"i2t_exact_r1:              {i2t_exact_r1 * 100:.2f}% (target: {v3_metrics['i2t_exact_r1'] * 100:.2f}%)")
        print(f"i2t_exact_r5:              {i2t_exact_r5 * 100:.2f}% (target: {v3_metrics['i2t_exact_r5'] * 100:.2f}%)")
        print(f"i2t_species_r1:            {i2t_species_r1 * 100:.2f}% (target: {v3_metrics['i2t_species_r1'] * 100:.2f}%)")
        print(f"i2t_species_r5:            {i2t_species_r5 * 100:.2f}% (target: {v3_metrics['i2t_species_r5'] * 100:.2f}%)")
        print(f"t2i_exact_r1:              {t2i_exact_r1 * 100:.2f}% (target: {v3_metrics['t2i_exact_r1'] * 100:.2f}%)")
        print(f"t2i_exact_r5:              {t2i_exact_r5 * 100:.2f}% (target: {v3_metrics['t2i_exact_r5'] * 100:.2f}%)")
        print(f"t2i_species_r1:            {t2i_species_r1 * 100:.2f}% (target: {v3_metrics['t2i_species_r1'] * 100:.2f}%)")
        print(f"t2i_species_r5:            {t2i_species_r5 * 100:.2f}% (target: {v3_metrics['t2i_species_r5'] * 100:.2f}%)")
        print(f"image_classifier_accuracy: {img_acc * 100:.2f}% (target: {v3_metrics['image_classifier_accuracy'] * 100:.2f}%)")
        print(f"text_classifier_accuracy:  {txt_acc * 100:.2f}% (target: {v3_metrics['text_classifier_accuracy'] * 100:.2f}%)")

if __name__ == '__main__':
    main()
