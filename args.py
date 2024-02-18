import argparse


def get_args():
    '''Get arguments from command line
    
    Returns:
        args (argparse.Namespace): parsed arguments
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='whether to train or test a model')
    parser.add_argument('--data_dir', default='mvtec_anomaly_detection', help='root directory containing data')
    parser.add_argument('--model_dir', default='model', help='directory where model is to be saved/retrieved')
    parser.add_argument('--results_dir', default='results', help='directory where results are saved')
    parser.add_argument('--category', default='carpet', choices=['carpet',
                                                                 'grid',
                                                                 'leather',
                                                                 'tile',
                                                                 'wood',
                                                                 'bottle',
                                                                 'cable',
                                                                 'capsule',
                                                                 'hazelnut',
                                                                 'metal_nut',
                                                                 'pill',
                                                                 'screw',
                                                                 'toothbrush',
                                                                 'transistor',
                                                                 'zipper'])
    parser.add_argument('--size', default=256, type=int, help='resize dimension of images')
    parser.add_argument('--device', default='cuda', help='device to train the model on')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs to train the model for')
    parser.add_argument('--lr', default=0.4, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4,  type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9,  type=float, help='momentum')
    parser.add_argument('--alphas', nargs=3, default=[1., 1., 1.], type=float, help='weights of feature maps for computation of distillation loss')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--mean', nargs=3, default=[0.485, 0.456, 0.406], type=float, help='mean for image normalization')
    parser.add_argument('--std', nargs=3, default=[0.229, 0.224, 0.225], type=float, help='standard deviation for image normalization')

    args = parser.parse_args()
    return args