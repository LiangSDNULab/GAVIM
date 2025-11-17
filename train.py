import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import evaluate
from utils import gaussian_kl, compute_maximum_mean_discrepancy, volume_computation, SimKernel, Smooth_P
import numpy as np
from sklearn.cluster import KMeans
from tqdm import trange
from torch.utils.data import DataLoader


def pretrain(model, pre_optimizer, pre_scheduler, imv_loader, args):
    print('Pre-Training......')
    eval_data = copy.deepcopy(imv_loader.dataset.data_list)
    eval_mask = copy.deepcopy(imv_loader.dataset.mask_list)
    for v in range(args.num_views):
        eval_data[v] = torch.tensor(eval_data[v], dtype=torch.float32).to(args.device)
        eval_mask[v] = torch.tensor(eval_mask[v], dtype=torch.float32).to(args.device)
    eval_labels = imv_loader.dataset.labels

    if args.likelihood == 'Bernoulli':
        likelihood_fn = nn.BCEWithLogitsLoss(reduction='none')
    else:
        likelihood_fn = nn.MSELoss(reduction='none')

    t = trange(args.pre_epochs, leave=True)

    model.train()
    for epoch in t:
        epoch_loss = []
        for _, (batch_idx, batch_data, batch_mask) in enumerate(imv_loader):
            pre_optimizer.zero_grad()
            batch_data = [sv_d.to(args.device) for sv_d in batch_data]
            batch_mask = [sv_m.to(args.device) for sv_m in batch_mask]
            z_sample, _, aggregated_mu, aggregated_var, xr_list = model(batch_data, batch_mask)

            kl_z = gaussian_kl(aggregated_mu, aggregated_var)
            d_z = compute_maximum_mean_discrepancy(z_sample)

            rec_term = []
            for v in range(args.num_views):
                sv_rec = torch.sum(likelihood_fn(xr_list[v], batch_data[v]), dim=1)
                exist_rec = sv_rec * batch_mask[v].squeeze()
                view_rec_loss = torch.mean(exist_rec)
                rec_term.append(view_rec_loss)
            rec_loss = sum(rec_term)

            elbo_loss = rec_loss + (1 - args.gamma) * kl_z + (args.gamma + args.lamda - 1) * d_z

            epoch_loss.append(elbo_loss.item())
            elbo_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            pre_optimizer.step()
        pre_scheduler.step()
        overall_loss = sum(epoch_loss) / len(epoch_loss)

        t.set_description(f'Epoch {epoch:>3}/{args.epochs}  Loss:{overall_loss:.2f}')
        t.refresh()

    try:
        with torch.no_grad():
            learn_similarity = SimKernel(args.metric, args.k_neighbor)
            aggregated_mu, aggregated_var = model.inference_z(eval_data, eval_mask)
            z_sample = model.sampling_fn(aggregated_mu, aggregated_var)
            similarity = learn_similarity(z_sample)
            z_similarity = Smooth_P(similarity, args.k)

            kmeans = KMeans(n_clusters=int(args.class_num), random_state=0).fit(aggregated_mu.detach().cpu())
            z_y_pred = kmeans.labels_
            acc, nmi, ari = evaluate(eval_labels, z_y_pred)
            print('pre kmeans: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(acc, nmi, ari))

            kmeans = KMeans(n_clusters=int(args.class_num), random_state=0).fit(z_sample.detach().cpu())
            z_y_pred = kmeans.labels_
            acc, nmi, ari = evaluate(eval_labels, z_y_pred)
            print('pre sample kmeans: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(acc, nmi, ari))

    except:
        print("fail")

    return z_similarity


def train(model, optimizer, scheduler, imv_loader, z_similarity, args):
    print('Training......')
    eval_data = copy.deepcopy(imv_loader.dataset.data_list)
    eval_mask = copy.deepcopy(imv_loader.dataset.mask_list)
    for v in range(args.num_views):
        eval_data[v] = torch.tensor(eval_data[v], dtype=torch.float32).to(args.device)
        eval_mask[v] = torch.tensor(eval_mask[v], dtype=torch.float32).to(args.device)
    eval_labels = imv_loader.dataset.labels

    if args.likelihood == 'Bernoulli':
        likelihood_fn = nn.BCEWithLogitsLoss(reduction='none')
    else:
        likelihood_fn = nn.MSELoss(reduction='none')

    learn_similarity = SimKernel(args.metric, args.k_neighbor)
    p_neighbor = torch.sum(z_similarity != 0, dim=1).max()

    t = trange(args.epochs, leave=True)

    for epoch in t:
        epoch_loss = []
        model.train()

        for _, (batch_idx, batch_data, batch_mask) in enumerate(imv_loader):
            optimizer.zero_grad()
            batch_data = [sv_d.to(args.device) for sv_d in batch_data]
            batch_mask = [sv_m.to(args.device) for sv_m in batch_mask]
            
            z_sample, vs_sample, aggregated_mu, aggregated_var, xr_list = model(batch_data, batch_mask)

            kl_z = gaussian_kl(aggregated_mu, aggregated_var)
            d_z = compute_maximum_mean_discrepancy(z_sample)

            rec_term = []
            for v in range(args.num_views):
                sv_rec = torch.sum(likelihood_fn(xr_list[v], batch_data[v]), dim=1)
                exist_rec = sv_rec * batch_mask[v].squeeze()
                view_rec_loss = torch.mean(exist_rec)
                rec_term.append(view_rec_loss)
            rec_loss = sum(rec_term)

            elbo_loss = rec_loss + (1 - args.gamma) * kl_z + (args.gamma + args.lamda - 1) * d_z

            volume = volume_computation(z_sample, vs_sample, batch_mask) / args.contrastive_temp
            targets = torch.linspace(0, z_sample.shape[0] - 1, z_sample.shape[0], dtype=int).to(args.device)
            c_loss = (F.cross_entropy(-volume, targets, label_smoothing=0.1) + F.cross_entropy(-volume.T, targets, label_smoothing=0.1)) / 2

            z_similarity_batch = z_similarity[batch_idx, :]
            _, nn_idx = torch.topk(z_similarity_batch, k=p_neighbor, dim=1, largest=True)

            n = len(batch_idx)
            z_P = torch.zeros(n, p_neighbor).to(args.device)
            row_idx = torch.arange(n, device=args.device).view(-1, 1).expand(-1, p_neighbor)
            column_idx = torch.arange(p_neighbor, device=args.device).view(1, -1).expand(n, -1)
            z_P[row_idx.flatten(), column_idx.flatten()] = z_similarity_batch[row_idx.flatten(), nn_idx.flatten()]
            z_P = z_P / (z_P.sum(dim=1, keepdim=True) + 1e-10)

            k_features = [[] for _ in range(args.num_views)]
            k_masks = [[] for _ in range(args.num_views)]
            for k_id in range(p_neighbor):
                for v in range(args.num_views):
                    k_masks[v].append(eval_mask[v][nn_idx[:, k_id]])
                    k_features[v].append(model.sv_encode(eval_data[v][nn_idx[:, k_id]], v))

            kl_term = []
            for v in range(args.num_views):
                k_feature = torch.stack(k_features[v], dim=1)
                k_mask = torch.cat(k_masks[v], dim=1)
                sv_mask = batch_mask[v] * k_mask
                v_similarity = learn_similarity(vs_sample[v], k_feature, k_mask)
                view_kl = F.kl_div(v_similarity.clamp(min=1e-10).log(), z_P, reduction='none')
                view_kl_loss = view_kl * sv_mask
                kl_term.append(view_kl_loss.mean())
            s_loss = sum(kl_term)

            batch_loss = elbo_loss + args.alpha * c_loss + args.beta * s_loss
            epoch_loss.append(batch_loss.item())
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        scheduler.step()
        overall_loss = sum(epoch_loss) / len(epoch_loss)

        with torch.no_grad():
            try:
                aggregated_mu, aggregated_var = model.inference_z(eval_data, eval_mask)
                z_sample = model.sampling_fn(aggregated_mu, aggregated_var)
                if (epoch + 1) % 10 == 0:
                    similarity = learn_similarity(z_sample)
                    z_similarity = Smooth_P(similarity, args.k)
                    p_neighbor = torch.sum(z_similarity != 0, dim=1).max()

                kmeans = KMeans(n_clusters=int(args.class_num), random_state=0).fit(aggregated_mu.detach().cpu())
                predict = kmeans.labels_
                acc, nmi, ari = evaluate(eval_labels, predict)

                t.set_description(f'Epoch {epoch + 1:>3}/{args.epochs}  Loss:{overall_loss:.2f}  elbo:{elbo_loss:.2f}  c:{c_loss:.2f}  s:{s_loss:.2f}  ACC:{acc * 100:.2f}  '
                        f'NMI:{nmi * 100:.2f}  ARI:{ari * 100:.2f}')
                t.refresh()
            except:
                print("nan")
                break

    try:
        kmeans = KMeans(n_clusters=int(args.class_num), random_state=0).fit(aggregated_mu.detach().cpu())
        z_y_pred = kmeans.labels_
        acc, nmi, ari = evaluate(eval_labels, z_y_pred)
        print('kmeans: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(acc, nmi, ari))

        z_sample = z_sample.detach().cpu()
        kmeans = KMeans(n_clusters=int(args.class_num), random_state=0).fit(z_sample.detach().cpu())
        z_y_pred = kmeans.labels_
        acc, nmi, ari = evaluate(eval_labels, z_y_pred)
        print('sample kmeans: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f}'.format(acc, nmi, ari))

    except:
        acc, nmi, ari = 0, 0, 0
        print("fail")

    return acc, nmi, ari

