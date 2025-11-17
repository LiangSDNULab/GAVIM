import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from DataSets import build_dataset
from Models import GAVIM
import numpy as np
import argparse
import random
from train import pretrain, train


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    test_record = {"ACC": [], "NMI": [], "ARI": []}
    setup_seed(args.seed)
    for t in range(1, args.test_times + 1):
        print(f'Test {t}')
        imv_dataset = build_dataset(args)
        imv_loader = DataLoader(imv_dataset, batch_size=args.batch_size, shuffle=True)
        model = GAVIM(args).to(args.device)

        pre_optimizer = optim.Adam(model.parameters(), lr=args.pre_learning_rate)
        pre_scheduler = optim.lr_scheduler.StepLR(pre_optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_factor)
        z_similarity = pretrain(model, pre_optimizer, pre_scheduler, imv_loader, args)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_factor)
        acc, nmi, ari = train(model, optimizer, scheduler, imv_loader, z_similarity, args)
        test_record["ACC"].append(acc)
        test_record["NMI"].append(nmi)
        test_record["ARI"].append(ari)
    print('Average ACC {:.2f} Average NMI {:.2f} Average ARI {:.2f}'.format(np.mean(test_record["ACC"]) * 100,
                                                                            np.mean(test_record["NMI"]) * 100,
                                                                            np.mean(test_record["ARI"]) * 100))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_epochs', type=int, default=100, help='pre-training epochs')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='training batch size')
    parser.add_argument('--pre_learning_rate', type=float, default=0.001, help='pre-training learning rate')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--z_dim', type=int, default=10, help='latent dimensions')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='StepLr_Step_size')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='StepLr_Gamma')
    parser.add_argument('--dataset', type=int, default=0, choices=range(5))
    parser.add_argument('--test_times', type=int, default=1)
    parser.add_argument('--missing_rate', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=100)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--lamda', type=float, default=5)
    parser.add_argument('--k_neighbor', type=int, default=3)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--contrastive_temp', type=float, default=0.05)
    parser.add_argument('--norm_type',type = str, default='standard')
    parser.add_argument('--metric',type = str, default='euclidean')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.dataset_dir_base = "datasets/"
    args.likelihood = 'Gaussian'


    if args.dataset == 0:
        args.dataset_name = 'HW'
        args.beta = 100
        args.lamda = 10
    elif args.dataset == 1:
        args.dataset_name = 'Scene-15'
        args.beta = 10
        args.lamda = 2
    elif args.dataset == 2:
        args.dataset_name = 'cub_googlenet_doc2vec_c10'
        args.beta = 100
        args.lamda = 5
    elif args.dataset == 3:
        args.dataset_name = 'Caltech7-5V'
        args.beta = 100
        args.lamda = 2
    elif args.dataset == 4:
        args.dataset_name = 'CCV'
        args.beta = 100
        args.lamda = 5

    print(f"Dataset : {args.dataset_name:<15} Missing rate : {args.missing_rate}")
    main(args)
