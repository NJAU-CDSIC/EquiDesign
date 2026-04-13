
import argparse
import os.path
from datasets import *

def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    from concurrent.futures import ProcessPoolExecutor
    from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
    from model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, EquiDesign

    scaler = torch.amp.GradScaler('cuda')
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    base_folder = time.strftime(args.path_for_outputs, time.localtime())
    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint
    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv",
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut,
        "HOMO"    : 0.70
    }

    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory': False,
                  'num_workers': 0}

    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train_set, valid_set, test_set = cath_dataset(
        args.batch_size,
        jsonl_file=args.jsonl_file,
        split_file=args.split_file,
    )
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    test_loader = torch.utils.data.DataLoader(test_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    model = EquiDesign(
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_encoder_layers,
        k_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
        equiformer_out_vector=getattr(args, 'equiformer_out_vector', 0),
    )
    model.to(device)

    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step']
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    with ProcessPoolExecutor(max_workers=12) as executor:
        q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        z = queue.Queue(maxsize=3)
        for i in range(3):
            q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            z.put_nowait(executor.submit(get_pdbs, test_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result()
        pdb_dict_test = z.get().result()

        loader_train = pdb_dict_train
        loader_valid = pdb_dict_valid
        loader_test = pdb_dict_test

        reload_c = 0
        best_test_acc = -1.0
        best_model_path = base_folder + "model_weights/best_model.pt"
        train_start_time = time.time()

        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    loader_train = pdb_dict_train
                    pdb_dict_valid = p.get().result()
                    loader_valid = pdb_dict_valid
                    q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                    p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                reload_c += 1
            for _, batch in enumerate(loader_train):
                start_batch = time.time()
                X, S, mask, ss = batch
                X = X.squeeze(0)
                S = S.squeeze(0)
                mask = mask.squeeze(0)
                ss = ss.squeeze(0)
                chain_M = torch.ones_like(S)
                X = X.to(device)
                S = S.to(device).long()
                mask = mask.to(device)
                ss = ss.to(device)
                chain_M = chain_M.to(device)
                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                mask_for_loss = mask * chain_M

                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        log_probs = model(X, S, mask, chain_M, ss)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    scaler.scale(loss_av_smoothed).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = model(X, S, mask, chain_M, ss)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    loss_av_smoothed.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
                    optimizer.step()

                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                for _, batch in enumerate(loader_valid):
                    X, S, mask, ss = batch
                    X = X.squeeze(0)
                    S = S.squeeze(0)
                    mask = mask.squeeze(0)
                    ss = ss.squeeze(0)
                    chain_M = torch.ones_like(S)
                    X = X.to(device)
                    S = S.to(device).long()
                    mask = mask.to(device)
                    ss = ss.to(device)
                    chain_M = chain_M.to(device)
                    log_probs = model(X, S, mask, chain_M, ss)
                    mask_for_loss = mask * chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)

            train_loss_ = np.format_float_positional(np.float32(train_loss), unique=False, precision=4)
            validation_loss_ = np.format_float_positional(np.float32(validation_loss), unique=False, precision=4)
            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=4)
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=4)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=4)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=4)

            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1 - t0), unique=False, precision=1)
            elapsed_total_sec = t1 - train_start_time
            elapsed_min = np.format_float_positional(np.float32(elapsed_total_sec / 60.0), unique=False, precision=2)
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, elapsed_min: {elapsed_min}, train_ppl: {train_perplexity_}, valid_ppl: {validation_perplexity_},train_loss:{train_loss_},valid_loss:{validation_loss_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, elapsed_min: {elapsed_min}, train_ppl: {train_perplexity_}, valid_ppl: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')

            checkpoint_filename_last = base_folder + 'model_weights/epoch_last.pt'
            torch.save({
                'epoch': e + 1,
                'step': total_step,
                'num_edges': args.num_neighbors,
                'noise_level': args.backbone_noise,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
            }, checkpoint_filename_last)

            if (e + 1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder + 'model_weights/epoch{}_step{}.pt'.format(e + 1, total_step)
                torch.save({
                    'epoch': e + 1,
                    'step': total_step,
                    'num_edges': args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                }, checkpoint_filename)

            model.eval()
            with torch.no_grad():
                test_sum, test_weights = 0., 0.
                test_acc = 0.
                for _, batch in enumerate(loader_test):
                    X, S, mask, ss = batch
                    X = X.squeeze(0)
                    S = S.squeeze(0)
                    mask = mask.squeeze(0)
                    ss = ss.squeeze(0)
                    chain_M = torch.ones_like(S)
                    X = X.to(device)
                    S = S.to(device).long()
                    mask = mask.to(device)
                    ss = ss.to(device)
                    chain_M = chain_M.to(device)
                    log_probs = model(X, S, mask, chain_M, ss)
                    mask_for_loss = mask * chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    test_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    test_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    test_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                test_loss = test_sum / test_weights
                test_accuracy = test_acc / test_weights
                test_perplexity = np.exp(test_loss)
                test_loss_ = np.format_float_positional(np.float32(test_loss), unique=False, precision=4)
                test_perplexity_ = np.format_float_positional(np.float32(test_perplexity), unique=False, precision=4)
                test_accuracy_ = np.format_float_positional(np.float32(test_accuracy), unique=False, precision=4)
                with open(logfile, 'a') as f:
                    f.write(f'epoch: {e+1}, step: {total_step}, elapsed_min: {elapsed_min}, test_loss: {test_loss_}, test_perplexity: {test_perplexity_}, test_acc: {test_accuracy_}\n')
                print(f'epoch: {e+1}, step: {total_step}, test_loss: {test_loss_}, test_perplexity: {test_perplexity_}, test_acc: {test_accuracy_}')

                if test_accuracy > best_test_acc:
                    best_test_acc = test_accuracy
                    torch.save({
                        'epoch': e + 1,
                        'step': total_step,
                        'num_edges': args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        'best_test_acc': best_test_acc,
                    }, best_model_path)
                    print(f"[BEST MODEL] Epoch {e+1}, step {total_step}, test_acc={best_test_acc:.4f}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data")
    argparser.add_argument("--jsonl_file", type=str, default="../data/chain_set.jsonl", help="path to CATH chain_set.jsonl (e.g. for CATH 4.4)")
    argparser.add_argument("--split_file", type=str, default="../data/chain_set_splits.json", help="path to train/val/test splits JSON")
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_equiformer", help="path for logs and model weights (EquiFormer)")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=800, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers")
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")
    argparser.add_argument("--dropout", type=float, default=0.32, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.00, help="amount of noise added to backbone during training")
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
    argparser.add_argument("--equiformer_out_vector", type=int, default=0, help="EquiFormer block vector channels (0=scalar only)")
    args = argparser.parse_args()
    main(args)
