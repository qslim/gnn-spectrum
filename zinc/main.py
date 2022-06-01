import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
from tqdm import tqdm
### importing OGB
# from ogb.graphproppred import Evaluator, collate_dgl

import sys
sys.path.append('..')

from model import Net
from utils.config import process_config, get_args
from utils.lr import MultiStepLRWarmUp
from data_preparation import MoleculeDataset


def train(model, device, loader, optimizer):
    model.train()
    loss_all = 0

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Train iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop('feat')
        edge_attr = bg.edata.pop('feat')
        bases = bg.edata.pop('bases')
        labels = labels.to(device)

        # (#0515)
        pred = model(bg, x, edge_attr, bases)
        optimizer.zero_grad()

        loss = F.l1_loss(pred, labels)
        loss.backward()
        optimizer.step()
        loss_all = loss_all + loss.detach().item()
    return loss_all / len(loader)


def eval(model, device, loader):
    model.eval()
    total_mae = 0

    with torch.no_grad():
        for step, (bg, labels) in enumerate(tqdm(loader, desc="Eval iteration")):
            bg = bg.to(device)
            x = bg.ndata.pop('feat')
            edge_attr = bg.edata.pop('feat')
            bases = bg.edata.pop('bases')
            labels = labels.to(device)

            # (#0515)
            pred = model(bg, x, edge_attr, bases)
            total_mae += F.l1_loss(pred, labels).detach().item()

        # (#0515)
        acc = total_mae / (step + 1)
        # acc = 1.0 * total_mae / step

    return acc


import time
def main():
    args = get_args()
    config = process_config(args)
    print(config)

    if config.get('seeds') is not None:
        for seed in config.seeds:
            config.seed = seed
            config.time_stamp = int(time.time())
            print(config)

            run_with_given_seed(config)
    else:
        run_with_given_seed(config)


def run_with_given_seed(config):
    if config.get('seed') is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### automatic dataloading and splitting
    dataset = MoleculeDataset(name=config.dataset_name, config=config)

    print("Bases total: {}".format(dataset.train.graph_lists[0].edata['bases'].shape[1]))

    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    train_loader = DataLoader(trainset, batch_size=config.hyperparams.batch_size, shuffle=True,
                              num_workers=config.num_workers, collate_fn=dataset.collate)
    valid_loader = DataLoader(valset, batch_size=config.hyperparams.batch_size, shuffle=False,
                              num_workers=config.num_workers, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=config.hyperparams.batch_size, shuffle=False,
                             num_workers=config.num_workers, collate_fn=dataset.collate)

    model = Net(config.architecture, num_tasks=1,
                num_basis=dataset.train.graph_lists[0].edata['bases'].shape[1],
                shared=config.get('filter', '') == 'shd').to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.AdamW(model.parameters(), lr=config.hyperparams.learning_rate,
                            weight_decay=config.hyperparams.weight_decay)
    scheduler = MultiStepLRWarmUp(optimizer, milestones=config.hyperparams.milestones,
                                  gamma=config.hyperparams.decay_rate,
                                  num_warm_up=config.hyperparams.warmup_epochs,
                                  init_lr=config.hyperparams.learning_rate)

    valid_curve = []
    test_curve = []
    train_curve = []
    trainL_curve = []

    writer = SummaryWriter(config.directory)

    ts_fk_algo_hp = str(config.time_stamp) + '_' \
                    + str(config.commit_id[0:7]) + '_' \
                    + str(config.get('filter', '')) \
                    + str(config.basis) \
                    + 'E' + str(config.epsilon) \
                    + 'P' + str(config.power) + '_' \
                    + str(config.architecture.pooling) + '_' \
                    + str(config.architecture.layers) + '_' \
                    + str(config.architecture.hidden) + '_' \
                    + str(config.hyperparams.learning_rate) + '_' \
                    + str(config.hyperparams.warmup_epochs) \
                    + str(config.hyperparams.milestones) + '_' \
                    + str(config.hyperparams.decay_rate) + '_' \
                    + 'B' + str(config.hyperparams.batch_size) \
                    + 'S' + str(config.seed) \
                    + 'W' + str(config.get('num_workers', 'na'))

    cur_epoch = 0
    if config.get('resume_train') is not None:
        print("Loading model from {}...".format(config.resume_train), end=' ')
        checkpoint = torch.load(config.resume_train)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        cur_epoch = checkpoint['epoch']
        cur_loss = checkpoint['loss']
        lr = checkpoint['lr']
        print("Model loaded.")

        print("Epoch {} evaluating...".format(cur_epoch))
        train_perf = eval(model, device, train_loader)
        valid_perf = eval(model, device, valid_loader)
        test_perf = eval(model, device, test_loader)

        print('Train:', train_perf,
              'Validation:', valid_perf,
              'Test:', test_perf,
              'Train loss:', cur_loss,
              'lr:', lr)

        train_curve.append(train_perf)
        valid_curve.append(valid_perf)
        test_curve.append(test_perf)
        trainL_curve.append(cur_loss)

        writer.add_scalars('traP', {ts_fk_algo_hp: train_perf}, cur_epoch)
        writer.add_scalars('valP', {ts_fk_algo_hp: valid_perf}, cur_epoch)
        writer.add_scalars('tstP', {ts_fk_algo_hp: test_perf}, cur_epoch)
        writer.add_scalars('traL', {ts_fk_algo_hp: cur_loss}, cur_epoch)
        writer.add_scalars('lr',   {ts_fk_algo_hp: lr}, cur_epoch)

    best_val = 10000.0
    for epoch in range(cur_epoch + 1, config.hyperparams.epochs + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        print("Epoch {} training...".format(epoch))
        train_loss = train(model, device, train_loader, optimizer)
        scheduler.step()

        print('Evaluating...')
        train_perf = eval(model, device, train_loader)
        valid_perf = eval(model, device, valid_loader)
        test_perf = eval(model, device, test_loader)

        # print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        print('Train:', train_perf,
              'Validation:', valid_perf,
              'Test:', test_perf,
              'Train loss:', train_loss,
              'lr:', lr)

        train_curve.append(train_perf)
        valid_curve.append(valid_perf)
        test_curve.append(test_perf)
        trainL_curve.append(train_loss)

        writer.add_scalars('traP', {ts_fk_algo_hp: train_perf}, epoch)
        writer.add_scalars('valP', {ts_fk_algo_hp: valid_perf}, epoch)
        writer.add_scalars('tstP', {ts_fk_algo_hp: test_perf}, epoch)
        writer.add_scalars('traL', {ts_fk_algo_hp: train_loss}, epoch)
        writer.add_scalars('lr',   {ts_fk_algo_hp: lr}, epoch)

        if config.get('checkpoint_dir') is not None:
            filename_header = str(config.commit_id[0:7]) + '_' \
                       + str(config.time_stamp) + '_' \
                       + str(config.dataset_name)
            if valid_perf < best_val:
                best_val = valid_perf
                filename = filename_header + 'best.tar'
            else:
                filename = filename_header + 'curr.tar'

            print("Saving model as {}...".format(filename), end=' ')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': train_loss,
                        'lr': lr},
                       os.path.join(config.checkpoint_dir, filename))
            print("Model saved.")

    writer.close()

    best_val_epoch = np.argmin(np.array(valid_curve))
    best_train = min(train_curve)

    print('Finished test: {}, Validation: {}, epoch: {}, best train: {}, best loss: {}'
          .format(test_curve[best_val_epoch], valid_curve[best_val_epoch],
                  best_val_epoch, best_train, min(trainL_curve)))


if __name__ == "__main__":
    main()
