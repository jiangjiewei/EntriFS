import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, f1_score
from tqdm import tqdm

from config import T_SAMPLES
from common import enable_dropout


@torch.no_grad()
def mc_dropout_predict(model, dataloader, device, t_samples=T_SAMPLES):
    model.eval()
    enable_dropout(model)
    records = []
    for images, labels, img_ids in tqdm(dataloader, desc='MC-Dropout', leave=False):
        images = images.to(device)
        probs = [torch.softmax(model(images), 1) for _ in range(t_samples)]
        mean_probs = torch.stack(probs).mean(0).cpu().numpy()
        labels_np = labels.numpy()
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
        for i in range(images.size(0)):
            pred = int(mean_probs[i].argmax())
            records.append({
                'Image_ID': img_ids[i],
                'True_Label': int(labels_np[i]),
                'Pred_Class': pred,
                'Max_Confidence': float(mean_probs[i].max()),
                'Predictive_Entropy': float(entropy[i]),
                'Is_Correct': int(pred == labels_np[i]),
            })
    return pd.DataFrame(records)


def metrics(df, reject_rate=0.0):
    sub = df.sort_values('Predictive_Entropy').head(int(len(df) * (1 - reject_rate)))
    acc = sub['Is_Correct'].mean() * 100
    qwk = cohen_kappa_score(sub['True_Label'], sub['Pred_Class'], weights='quadratic')
    f1 = f1_score(sub['True_Label'], sub['Pred_Class'], average='macro') * 100
    return acc, qwk, f1, len(sub)
