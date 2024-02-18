import os
import torch
from model import STFPM
from train import train
from evaluate import evaluate, get_images
from args import get_args


if __name__ == '__main__':
    args = get_args()
    stfpm = STFPM(backbone=args.backbone, layers=args.layers, size=args.size).to(args.device)

    if args.mode == 'train':
        train(model=stfpm, args=args)
        evaluate(model=stfpm, args=args)
    else:
        print('Loading model...')
        model_path = os.path.join(args.model_dir, f'student_{args.category}.pt')
        if not os.path.exists(model_path):
            raise Exception(f'Model not found at {model_path}. Check model path.')
        stfpm.student.load_state_dict(torch.load(model_path, map_location=args.device))
        print('Model loaded.\n')
        
        evaluate(model=stfpm, args=args)
        get_images(model=stfpm, args=args)