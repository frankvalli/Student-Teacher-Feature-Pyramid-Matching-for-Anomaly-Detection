import os
import glob
import argparse
import time

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score
from dataset import AnomalyDataset, get_transforms


def evaluate(model: nn.Module, args: argparse.Namespace) -> None:
    '''Evaluate the STFPM model

    Args:
        model (nn.Module): model to evaluate
        args (argparse.Namespace): parsed arguments
    '''
    transform = get_transforms(size=args.size, mean=args.mean, std=args.std)
    root_dir = os.path.join(os.path.join(args.data_dir, args.category), 'test')
    test_dataset = AnomalyDataset(root_dir=root_dir, transform=transform)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model.eval()
    gts_image = []
    preds_image = []
    gts_pixel = []
    preds_pixel = []

    print('Evaluating...\n')

    start_time = time.time()

    for sample in loader:
        img = sample['image'].to(args.device)
        mask = sample['mask'].to(args.device)
        label = sample['label'].to(args.device)

        gts_image.extend(label.tolist())
        gts_pixel.extend(torch.flatten(mask).detach().cpu().numpy())

        with torch.no_grad():
            anomaly_map, _ = model.get_anomaly_map(img)

        preds_image.extend(torch.amax(anomaly_map, dim=(-1, -2)).detach().cpu().numpy())
        preds_pixel.extend(torch.flatten(anomaly_map).detach().cpu().numpy())

    auc_image = roc_auc_score(gts_image, preds_image)
    auc_pixel = roc_auc_score(gts_pixel, preds_pixel)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'Elapsed time: {elapsed_time:.3f}')
    print(f'Image level AUC: {auc_image:.3f}')
    print(f'Pixel level AUC: {auc_pixel:.3f}\n')


def norm_amap(amap: torch.Tensor) -> torch.Tensor:
    '''Normalizes anomaly map between 0 and 1
    
    Args:
        amap (torch.Tensor): anomaly map

    Returns:
        torch.Tensor: normalized anomaly map
    '''
    px_min, px_max = amap.min(), amap.max()
    return (amap - px_min) / (px_max - px_min)


def inverse_norm(image: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    '''Gets original image from normalized one
    
    Args:
        image (torch.Tensor): normalized image
        mean (list[float]): means used to normalize image
        std (list[float]): stds used to normalize image

    Returns:
        torch.Tensor: unnormalized image
    '''
    mean_tensor = torch.tensor(mean, device=image.device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=image.device).view(1, -1, 1, 1)
    return image * std_tensor + mean_tensor


def get_images(model: nn.Module, args: argparse.Namespace) -> None:
    '''Applies model to dataset and gets anomaly maps
    
    Args:
        model (nn.Module): model to evaluate
        args (argparse.Namespace): parsed arguments
    '''
    model.eval()
    alpha = 0.5
    beta = 1. - alpha
    transform = get_transforms(size=args.size, mean=args.mean, std=args.std)
    start_time = time.time()

    print('Getting result images...\n')

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    category_dir = os.path.join(args.data_dir, args.category)
    img_dir = os.path.join(category_dir, 'test')
    defects = os.listdir(img_dir)
    defects.remove('good')

    for defect in defects:
        defect_dir = os.path.join(args.results_dir, args.category)
        if not os.path.exists(defect_dir):
            os.mkdir(defect_dir)
        defect_dir = os.path.join(defect_dir, defect)
        if not os.path.exists(defect_dir):
            os.mkdir(defect_dir)

        images = glob.glob(os.path.join(img_dir, defect)+'/*.png')

        for idx, img_path in enumerate(images):
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(args.device)

            with torch.no_grad():
                anomaly_map, anomaly_map_list = model.get_anomaly_map(img)
            anomaly_map_list.append(anomaly_map)

            img = inverse_norm(img, args.mean, args.std)
            img_numpy = np.uint8(img.squeeze().detach().cpu() * 255)
            img_numpy = np.transpose(img_numpy, (1, 2, 0))
            img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)

            for a, amap in enumerate(anomaly_map_list):
                amap = norm_amap(amap.squeeze())
                heatmap = cv2.applyColorMap(np.uint8((amap * 255).detach().cpu()), cv2.COLORMAP_JET)
                overlap = cv2.addWeighted(img_numpy, alpha, heatmap, beta, 0)
                cv2.imwrite(os.path.join(defect_dir, f'{idx:03d}_amap{a+1}.png'), overlap)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time:.3f}')
    print('Finished.')