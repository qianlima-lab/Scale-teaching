import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import os
import random
from math import inf
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data as data
from scipy import stats
from sklearn.metrics import accuracy_score
from scipy.io.arff import loadarff
from model_scale import FCN, Classifier


def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


def build_model(args):
    if args.backbone == 'fcn':
        model = FCN(args.num_classes, args.input_size)

    if args.classifier == 'linear':
        classifier = Classifier(args.classifier_input, args.num_classes)

    return model, classifier


def build_loss(args):
    if args.loss == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif args.loss == 'reconstruction':
        return nn.MSELoss()


def transfer_labels(labels):
    indicies = np.unique(labels)
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indicies)[0][0]
        labels[i] = new_label

    return labels


def build_dataset_pt(args):
    data_path = f"../data/ts_noise_data/{args.dataset}"
    train_dataset_dict = torch.load(os.path.join(data_path, "train.pt"))
    train_dataset = train_dataset_dict["samples"].numpy()  # (num_size, num_dimensions, series_length)
    train_target = train_dataset_dict["labels"].numpy()
    num_classes = len(np.unique(train_dataset_dict["labels"].numpy(), return_counts=True)[0])
    train_target = transfer_labels(train_target)

    test_dataset_dict = torch.load(os.path.join(data_path, "test.pt"))
    test_dataset = test_dataset_dict["samples"].numpy()  # (num_size, num_dimensions, series_length)
    test_target = test_dataset_dict["labels"].numpy()
    test_target = transfer_labels(test_target)

    return train_dataset, train_target, test_dataset, test_target, num_classes


def build_dataset_ucr(args):
    data_path = f"../data/UCRArchive_2018"
    train = pd.read_csv(os.path.join(data_path, args.dataset, args.dataset + '_TRAIN.tsv'), sep='\t', header=None)
    train_x = train.iloc[:, 1:]
    train_target = train.iloc[:, 0]

    test = pd.read_csv(os.path.join(data_path, args.dataset, args.dataset + '_TEST.tsv'), sep='\t', header=None)
    test_x = test.iloc[:, 1:]
    test_target = test.iloc[:, 0]

    train_dataset = train_x.to_numpy(dtype=np.float32)
    train_target = train_target.to_numpy(dtype=np.float32)
    test_dataset = test_x.to_numpy(dtype=np.float32)
    test_target = test_target.to_numpy(dtype=np.float32)

    num_classes = len(np.unique(train_target))
    train_target = transfer_labels(train_target)
    test_target = transfer_labels(test_target)

    train_dataset = torch.unsqueeze(torch.from_numpy(train_dataset), 1)
    test_dataset = torch.unsqueeze(torch.from_numpy(test_dataset), 1)
    train_dataset = train_dataset.numpy()
    test_dataset = test_dataset.numpy()

    print("train shape = ", train_dataset.shape, ", test shape = ", test_dataset.shape)

    ind = np.where(np.isnan(train_dataset))
    col_mean = np.nanmean(train_dataset, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_dataset[ind] = np.take(col_mean, ind[1])

    ind_test = np.where(np.isnan(test_dataset))
    test_dataset[ind_test] = np.take(col_mean, ind_test[1])

    train_dataset, train_target = shuffler_dataset(train_dataset, train_target)
    test_dataset, test_target = shuffler_dataset(test_dataset, test_target)

    return train_dataset, train_target, test_dataset, test_target, num_classes


def build_dataset_uea(args):
    data_path = f"../data/Multivariate2018_arff"
    train_data = loadarff(os.path.join(data_path, args.dataset, args.dataset + '_TRAIN.arff'))[0]
    test_data = loadarff(os.path.join(data_path, args.dataset, args.dataset + '_TEST.arff'))[0]

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)

    train_dataset = train_X.transpose(0, 2, 1)
    train_target = train_y
    test_dataset = test_X.transpose(0, 2, 1)
    test_target = test_y

    num_classes = len(np.unique(train_target))
    train_target = transfer_labels(train_target)
    test_target = transfer_labels(test_target)

    ind = np.where(np.isnan(train_dataset))
    col_mean = np.nanmean(train_dataset, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_dataset[ind] = np.take(col_mean, ind[1])

    ind_test = np.where(np.isnan(test_dataset))
    test_dataset[ind_test] = np.take(col_mean, ind_test[1])

    train_dataset, train_target = shuffler_dataset(train_dataset, train_target)
    test_dataset, test_target = shuffler_dataset(test_dataset, test_target)

    return train_dataset, train_target, test_dataset, test_target, num_classes


def shuffler_dataset(x_train, y_train):
    indexes = np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    return x_train, y_train


def evaluate_scale_flow_acc(val_loader, model_list, classifier_list, loss_list, scale_list, device):
    target_true = []
    target_pred = []
    val_loss = 0
    num_val_samples = 0
    for data, target in val_loader:
        with torch.no_grad():
            up_scale_embed = None
            index_s = 0
            for scale_value in scale_list:
                if index_s == 0:
                    pred_embed = model_list[index_s](downsample_torch(data, sample_rate=scale_value, device=device))
                    up_scale_embed = pred_embed

                    if index_s == (len(scale_list) - 1):  ## len(scale_list) = 1, this is a single scale
                        val_pred = classifier_list[index_s](pred_embed)
                        step_loss = loss_list[index_s](val_pred, target)

                        val_loss += step_loss.item()
                        target_true.append(target.cpu().numpy())
                        target_pred.append(torch.argmax(val_pred.data, axis=1).cpu().numpy())
                        num_val_samples = num_val_samples + len(target)

                else:
                    pred_embed = model_list[index_s](downsample_torch(data, sample_rate=scale_value, device=device),
                                                     scale_x=up_scale_embed, is_head=True)
                    up_scale_embed = pred_embed
                    val_pred = classifier_list[index_s](pred_embed)
                    step_loss = loss_list[index_s](val_pred, target)

                    if index_s == (len(scale_list) - 1):
                        val_loss += step_loss.item()
                        target_true.append(target.cpu().numpy())
                        target_pred.append(torch.argmax(val_pred.data, axis=1).cpu().numpy())
                        num_val_samples = num_val_samples + len(target)
                index_s = index_s + 1

    target_true = np.concatenate(target_true)
    target_pred = np.concatenate(target_pred)

    return val_loss / num_val_samples, accuracy_score(target_true, target_pred)


def shuffler(x_train, y_train):
    indexes = np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    return x_train, y_train


class TimeDataset(data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        # self.dataset = np.expand_dims(self.dataset, 1)
        # print("dataset shape = ", dataset.shape)
        if len(self.dataset.shape) == 2:
            self.dataset = torch.unsqueeze(self.dataset, 1)
        # print("dataset shape = ", self.dataset.shape)
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)


def downsample(x_data, sample_rate):
    """
     Takes a batch of sequences with [batch size, channels, seq_len] and down-samples with sample
     rate k. hence, every k-th element of the original time series is kept.
    """
    last_one = 0
    if x_data.shape[2] % sample_rate > 0:
        last_one = 1

    new_length = int(np.floor(x_data.shape[2] / sample_rate)) + last_one
    output = np.zeros((x_data.shape[0], x_data.shape[1], new_length))
    output[:, :, range(new_length)] = x_data[:, :, [i * sample_rate for i in range(new_length)]]

    return output


def downsample_torch(x_data, sample_rate, device):
    """
     Takes a batch of sequences with [batch size, channels, seq_len] and down-samples with sample
     rate k. hence, every k-th element of the original time series is kept.
    """
    last_one = 0
    if x_data.shape[2] % sample_rate > 0:
        last_one = 1

    new_length = int(np.floor(x_data.shape[2] / sample_rate)) + last_one
    output = torch.zeros((x_data.shape[0], x_data.shape[1], new_length)).to(device)
    output[:, :, range(new_length)] = x_data[:, :, [i * sample_rate for i in range(new_length)]]

    return output


def get_graph_nearind(data_embed, topk=20, sigma=0.25):
    eps = np.finfo(float).eps

    # step: graph construction
    emb_all = data_embed / (sigma + eps)  # n*d
    emb1 = torch.unsqueeze(emb_all, 1)  # n*1*d
    emb2 = torch.unsqueeze(emb_all, 0)  # 1*n*d
    w = ((emb1 - emb2) ** 2).mean(2)  # n*n*d -> n*n
    w = torch.exp(-w / 2)

    ## keep top-k values
    topk, indices = torch.topk(w, topk)

    # print("topk = ", topk, ", indices = ", indices)

    return topk, indices.cpu()


def get_one_near_ind(near_inds, y_mask):
    neighbors_idxs = []
    for idxs in near_inds:
        y_idxs = y_mask[idxs]
        # consist_label = idxs[1:]
        consist_label = idxs[y_idxs == 0][1:]  ## 0 is noisy, 1 is clean
        # print("consist_label = ", consist_label)
        if len(consist_label) == 0:
            neighbors_idxs.append(idxs[0].item())
            continue
        neighbor_index = np.random.choice(consist_label, 1)[0]
        neighbors_idxs.append(neighbor_index)

    return neighbors_idxs


def adjust_learning_rate(alpha_plan, optimizer, epoch_now):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch_now]


def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std=0.1, seed=42):
    # n -> noise_rate
    # dataset
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size
    # norm_std -> default 0.1
    # seed -> random_seed

    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)

    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        # print("x.shape = ", x.shape, ", x.view(1, -1).shape = ", x.view(1, -1).shape, W.shape, W[y].shape)
        # print(", flip_rate.shape = ", flip_rate.shape, ", y = ", y)
        A = x.view(1, -1).mm(W[y]).squeeze(0)

        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()

    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    print(f'noise rate = {(new_label != np.array(labels.cpu())).mean()}')

    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1
        #
    print('****************************************')
    print('following is flip percentage:')

    for i in range(label_num):
        sum_i = sum(record[i])
        for j in range(label_num):
            if i != j:
                print(f"{record[i][j] / sum_i: .2f}", end='\t')
            else:
                print(f"{record[i][j] / sum_i: .2f}", end='\t')
        print()

    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            for j in range(label_num):
                print(f"{a[j]:.2f}", end="\t")
            print()
            cnt += 1
        if cnt >= 10:
            break
    print(P)
    return np.array(new_label)


def flip_label(dataset, target, ratio, args, pattern=0):
    """
    Induce label noise by randomly corrupting labels
    :param target: list or array of labels
    :param ratio: float: noise ratio
    :param pattern: flag to choose which type of noise.
            0 or mod(pattern, #classes) == 0 = symmetric
            int = asymmetric
            -1 = flip
    :return:
    """
    assert 0 <= ratio < 1

    target = np.array(target).astype(int)
    label = target.copy()
    n_class = len(np.unique(label))

    if type(pattern) is int:
        if pattern == -1:
            # Instance
            num_classes = len(np.unique(target, return_counts=True)[0])
            data = torch.from_numpy(dataset).type(torch.FloatTensor)
            targets = torch.from_numpy(target).type(torch.FloatTensor).to(torch.int64)
            dataset_ = zip(data, targets)
            feature_size = dataset.shape[1] * dataset.shape[2]
            label = get_instance_noisy_label(n=ratio, dataset=dataset_, labels=targets, num_classes=num_classes,
                                             feature_size=feature_size, seed=args.random_seed)
        else:
            for i in range(label.shape[0]):
                # symmetric noise
                if (pattern % n_class) == 0:
                    p1 = ratio / (n_class - 1) * np.ones(n_class)
                    p1[label[i]] = 1 - ratio
                    label[i] = np.random.choice(n_class, p=p1)
                elif pattern == 1:
                    # Asymm
                    label[i] = np.random.choice([label[i], (target[i] + pattern) % n_class], p=[1 - ratio, ratio])

    elif type(pattern) is str:
        raise ValueError

    mask = np.array([int(x != y) for (x, y) in zip(target, label)])

    return label, mask


def evaluate_embed_loss_mask(val_loader, model, classifier, loss, remember_rate):
    embedding = []
    loss_all = []
    for data, target in val_loader:
        with torch.no_grad():
            val_pred = model(data)
            embedding.append(val_pred.cpu().detach().numpy())
            val_pred = classifier(val_pred)
            step_loss_all = loss(val_pred, target)
            loss_all.append(step_loss_all.cpu().detach().numpy())

    embedding = np.concatenate(embedding)
    loss_all = np.concatenate(loss_all)
    ind_1_sorted = np.argsort(loss_all)
    mask_loss = np.zeros(len(ind_1_sorted))
    for i in range(int(len(ind_1_sorted) * remember_rate)):
        mask_loss[ind_1_sorted[i]] = 1  ## 1 is samll loss (clean), 0 is big loss (noise)

    return torch.tensor(embedding), mask_loss


def construct_lp_graph(data_embed, y_label, mask_label, device, topk=20, sigma=0.25, alpha=0.99,
                       p_cutoff=0.99, num_real_class=2):
    eps = np.finfo(float).eps
    n, d = data_embed.shape[0], data_embed.shape[1]
    data_embed = data_embed
    emb_all = data_embed / (sigma + eps)  # n*d
    emb1 = torch.unsqueeze(emb_all, 1)  # n*1*d
    emb2 = torch.unsqueeze(emb_all, 0)  # 1*n*d
    w = ((emb1 - emb2) ** 2).mean(2)  # n*n*d -> n*n
    w = torch.exp(-w / 2)

    ## keep top-k values
    topk, indices = torch.topk(w, topk)
    mask = torch.zeros_like(w).to(device)
    mask = mask.scatter(1, indices, 1)
    mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, knn graph
    # mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, knn graph
    w = w * mask

    ## normalize
    d = w.sum(0)
    d_sqrt_inv = torch.sqrt(1.0 / (d + eps)).to(device)
    d1 = torch.unsqueeze(d_sqrt_inv, 1).repeat(1, n)
    d2 = torch.unsqueeze(d_sqrt_inv, 0).repeat(n, 1)
    s = d1 * w * d2

    # step2: label propagation, f = (i-\alpha s)^{-1}y
    y = torch.zeros(y_label.shape[0], num_real_class)
    for i in range(len(mask_label)):
        if mask_label[i] == 1:  ## 1 is clean label, 0 is noise
            y[i][int(y_label[i])] = 1
    f = torch.matmul(torch.inverse(torch.eye(n).to(device) - alpha * s + eps), y.to(device))
    all_knn_label = torch.argmax(f, 1).cpu().numpy()  ## all propagated pesudo label
    end_knn_label = torch.argmax(f, 1).cpu().numpy()

    class_counter = Counter(y_label)
    for i in range(num_real_class):
        class_counter[i] = 0
    for i in range(len(mask_label)):
        if mask_label[i] == 1:  ## 1 is clean label, 0 is noise
            end_knn_label[i] = y_label[i]  ## use clean label to replace pesudo label
        else:
            class_counter[all_knn_label[i]] += 1  ## use curriculm learning to select pesudo label

    classwise_acc = torch.zeros((num_real_class,)).to(device)
    for i in range(num_real_class):
        classwise_acc[i] = class_counter[i] / max(class_counter.values())
    pseudo_label = torch.softmax(f, dim=-1)  ##
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    cpl_mask = max_probs.ge(p_cutoff * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx])))

    end_clean_mask = np.zeros(len(mask_label))
    for i in range(len(mask_label)):
        if mask_label[i] == 1:  ## 1 is clean label, 0 is noise
            end_clean_mask[i] = 1
        else:
            if cpl_mask[i]:  ## the pesudo label is selected as a new clean label
                end_clean_mask[i] = 1

    return torch.tensor(end_knn_label).to(device).to(torch.int64), end_clean_mask


# KL divergence
def kl_divergence(p, q):
    return (p * ((p + 1e-10) / (q + 1e-10)).log()).sum(dim=1)


## Jensen-Shannon Divergence
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon, self).__init__()
        pass

    def forward(self, p, q):
        m = (p + q) / 2
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def calculate_scale_flow_jsd(val_loader, model_list, classifier_list, scale_list, num_class, num_samples, device):
    JS_dist = Jensen_Shannon()
    JSD = []

    for i in range(len(scale_list)):
        _jsd = torch.zeros(num_samples)
        JSD.append(_jsd)

    _start = 0
    for data, target in val_loader:
        with torch.no_grad():
            up_scale_embed = None
            index_s = 0
            for scale_value in scale_list:
                if index_s == 0:
                    pred_embed = model_list[index_s](downsample_torch(data, sample_rate=scale_value, device=device))
                    up_scale_embed = pred_embed
                    val_pred = classifier_list[index_s](pred_embed)

                    dist = JS_dist(val_pred, F.one_hot(target, num_classes=num_class))
                    # JSD[int(batch_idx * batch_size):int((batch_idx + 1) * batch_size)] = dist
                    JSD[index_s][_start: (_start + len(target))] = dist
                else:
                    pred_embed = model_list[index_s](downsample_torch(data, sample_rate=scale_value, device=device),
                                                     scale_x=up_scale_embed, is_head=True)
                    up_scale_embed = pred_embed
                    val_pred = classifier_list[index_s](pred_embed)
                    dist = JS_dist(val_pred, F.one_hot(target, num_classes=num_class))
                    # JSD[index_s].append(dist)
                    JSD[index_s][_start: (_start + len(target))] = dist
                index_s = index_s + 1

        _start = _start + len(target)

    return JSD


def get_clean_class_jsd_ind(jsd_all, remember_rate, class_num, target_label):
    '''
    :return: mask_jsd, 1 is clean, 0 is noise
    '''
    mask_jsd = np.zeros(len(jsd_all))

    for _cls in range(class_num):
        class_ind = np.where(target_label == _cls)[0]
        class_len = int(remember_rate * len(target_label) / class_num)
        jsd_class = np.argsort(jsd_all[class_ind])
        mask_jsd[class_ind[jsd_class[0:class_len]]] = 1  ## 1 is samll loss (clean), 0 is big loss (noise)

    return mask_jsd


def get_clean_loss_ind(loss_all, remember_rate):
    '''
    :param loss: numpy
    :param remember_rate: float, 1 - noise_rate
    :return: mask_loss, 1 is clean, 0 is noise
    '''
    ind_1_sorted = np.argsort(loss_all)
    mask_loss = np.zeros(len(ind_1_sorted))
    for i in range(int(len(ind_1_sorted) * remember_rate)):
        mask_loss[ind_1_sorted[i]] = 1  ## 1 is samll loss (clean), 0 is big loss (noise)

    return mask_loss


def get_clean_class_loss_ind(loss_all, remember_rate, class_num, predict_label):
    '''
    :param loss: numpy
    :param remember_rate: float, 1 - noise_rate
    :return: mask_loss, 1 is clean, 0 is noise
    '''
    mask_loss = np.zeros(len(loss_all))
    predict_label = np.array(predict_label).astype(int)

    for i in range(class_num):
        ind_class_i = np.where(predict_label == i)
        ind_class_i_loss = loss_all[ind_class_i[0]]
        ind_class_i_loss_sorted = np.argsort(ind_class_i_loss)
        if len(ind_class_i_loss_sorted) < 1:
            continue
        ind_class_i = ind_class_i[0][ind_class_i_loss_sorted]
        len_r = int(len(ind_class_i) * remember_rate)
        mask_loss[ind_class_i[:len_r]] = 1  ## 1 is samll loss (clean), 0 is big loss (noise)

    return mask_loss
