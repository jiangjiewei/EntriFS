import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes=5, dropout=0.5, pretrained=True):
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet50(weights=weights)
    except AttributeError:
        net = models.resnet50(pretrained=pretrained)
    in_f = net.fc.in_features
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    return net


def set_trainable(model, mode):
    """mode: fc_only | layer4_fc | layer34_fc | all"""
    for p in model.parameters():
        p.requires_grad = False
    if mode == 'fc_only':
        for p in model.fc.parameters():
            p.requires_grad = True
    elif mode == 'layer4_fc':
        for name, p in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                p.requires_grad = True
    elif mode == 'layer34_fc':
        for name, p in model.named_parameters():
            if 'layer3' in name or 'layer4' in name or 'fc' in name:
                p.requires_grad = True
    elif mode == 'all':
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f'Unknown mode: {mode}')


def optimizer_for_mode(model, mode):
    import torch.optim as optim
    if mode == 'fc_only':
        return optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    if mode == 'layer4_fc':
        return optim.Adam([
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3},
        ], weight_decay=1e-4)
    if mode == 'layer34_fc':
        return optim.Adam([
            {'params': model.layer3.parameters(), 'lr': 5e-5},
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3},
        ], weight_decay=1e-4)
    if mode == 'all':
        return optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    raise ValueError(mode)
