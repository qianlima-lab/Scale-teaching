import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from scare_utils import downsample_torch, build_dataset_pt, build_dataset_ucr, build_dataset_uea, build_model, \
    set_seed, shuffler, flip_label, adjust_learning_rate, TimeDataset, build_loss, \
    construct_lp_graph, calculate_scale_flow_jsd, get_clean_class_jsd_ind, get_clean_loss_ind, get_clean_class_loss_ind, evaluate_scale_flow_acc

if __name__ == '__main__':  ##

    parser = argparse.ArgumentParser()

    # Base setup
    parser.add_argument('--backbone', type=str, default='fcn', help='encoder backbone, fcn')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='SleepEDF',
                        help='')  # ['HAR', 'SleepEDF', 'FD-A', 'unimib2']
    parser.add_argument('--archive', type=str, default='Four',
                        help='Four, UCR, UEA')
    parser.add_argument('--num_classes', type=int, default=0, help='number of class')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')

    # Label noise
    parser.add_argument('--label_noise_type', type=int, default=0,
                        help='0 is Sym, 1 is Asym, -1 is Instance')
    parser.add_argument('--label_noise_rate', type=float, default=0.5,
                        help='label noise ratio, sym: 0.2, 0.5, asym: 0.4, ins: 0.4')
    parser.add_argument('--warmup_epoch', type=int, default=30, help='30 or 50')
    parser.add_argument('--small_loss_criterion', type=int, default=1, help='1 is use the warm_up small loss, 0 is use the jsd small loss.')
    parser.add_argument('--scale_nums', type=int, default=3, help='3, 4, 5, 6')
    parser.add_argument('--scale_list', type=list, default=[1, 2, 4], help='')
    parser.add_argument('--knn_num', type=int, default=10, help='')
    parser.add_argument('--moment_alpha', type=float, default=0.9, help='')
    parser.add_argument('--epoch_decay_start', type=int, default=80)
    parser.add_argument('--epoch_correct_start', type=int, default=120)

    # training setup
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--epoch', type=int, default=200, help='training epoch')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    # classifier setup
    parser.add_argument('--classifier', type=str, default='linear', help='')
    parser.add_argument('--classifier_input', type=int, default=128, help='input dim of the classifiers')

    args = parser.parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)

    # define drop rate schedule
    rate_schedule = np.ones(args.epoch) * args.label_noise_rate
    rate_schedule[:args.warmup_epoch] = np.linspace(0, args.label_noise_rate, args.warmup_epoch)

    if args.archive == 'Four':
        train_dataset, train_target, test_dataset, test_target, num_classes = build_dataset_pt(args)
    elif args.archive == 'UCR':
        train_dataset, train_target, test_dataset, test_target, num_classes = build_dataset_ucr(args)
    elif args.archive == 'UEA':
        train_dataset, train_target, test_dataset, test_target, num_classes = build_dataset_uea(args)

    args.num_classes = num_classes
    args.seq_len = train_dataset.shape[2]
    args.input_size = train_dataset.shape[1]

    train_dataset, train_target = shuffler(train_dataset, train_target)

    alpha_plan = [args.lr] * args.epoch
    for i in range(args.epoch_decay_start, args.epoch):
        alpha_plan[i] = float(args.epoch - i) / (args.epoch - args.epoch_decay_start) * args.lr

    model_list = []
    classifier_list = []
    loss_list = []
    loss_sample_select_list = []
    optimizer_list = []
    for s in range(args.scale_nums):
        model, classifier = build_model(args)
        classifier = classifier.to(device)
        model = model.to(device)

        loss = build_loss(args).to(device)

        optimizer = torch.optim.Adam([{'params': model.parameters()},
                                      {'params': classifier.parameters()}],
                                     lr=args.lr)

        model_list.append(model)
        classifier_list.append(classifier)
        loss_list.append(loss)
        loss_sample_select_list.append(torch.nn.CrossEntropyLoss(reduce=False).to(device))
        optimizer_list.append(optimizer)

    train_time = 0.0
    t = time.time()

    if args.label_noise_rate > 0:
        train_target, mask_train_target = flip_label(dataset=train_dataset, target=train_target,
                                                     ratio=args.label_noise_rate, args=args
                                                     , pattern=args.label_noise_type)

    train_set = TimeDataset(torch.from_numpy(train_dataset).type(torch.FloatTensor).to(device),
                            torch.from_numpy(train_target).type(torch.FloatTensor).to(device).to(torch.int64))

    test_set = TimeDataset(torch.from_numpy(test_dataset).type(torch.FloatTensor).to(device),
                           torch.from_numpy(test_target).type(torch.FloatTensor).to(device).to(torch.int64))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

    up_outputs_list = []
    for _ in range(args.scale_nums):
        _output = torch.zeros(train_dataset.shape[0], 128).float().to(device)
        up_outputs_list.append(_output)

    test_end_five_accuracies = []

    for epoch in range(args.epoch):
        epoch_train_loss = 0
        epoch_train_acc = 0
        num_iterations = 0

        mask_clean = []
        if epoch > args.warmup_epoch:
            for s in range(args.scale_nums):
                model_list[s].eval()
                classifier_list[s].eval()

            if args.small_loss_criterion == 0:  ## use jsd small loss criterion
                JSD = Calculate_scale_flow_jsd(val_loader=train_loader,
                                               model_list=model_list,
                                               classifier_list=classifier_list,
                                               scale_list=args.scale_list,
                                               num_class=args.num_classes,
                                               num_samples=train_dataset.shape[0],
                                               device=device)

                for _jsd in JSD:
                    threshold = torch.mean(_jsd)
                    if threshold.item() > args.d_u:
                        threshold = threshold - (threshold - torch.min(_jsd)) / args.tau
                    _sr = torch.sum(_jsd < threshold).item() / train_dataset.shape[0]

                    _mask_clean = get_clean_class_jsd_ind(jsd_all=_jsd,
                                                          remember_rate=_sr,
                                                          class_num=args.num_classes,
                                                          target_label=train_target)

                    mask_clean.append(_mask_clean)

        for s in range(args.scale_nums):
            model_list[s].train()
            classifier_list[s].train()
            adjust_learning_rate(alpha_plan, optimizer_list[s], epoch)

        for x, y in train_loader:
            _start = num_iterations * args.batch_size
            _end = (num_iterations + 1) * args.batch_size
            if _end > train_dataset.shape[0]:
                _end = train_dataset.shape[0]

            if (_end - _start) <= 1:
                continue

            for s in range(args.scale_nums):
                optimizer_list[s].zero_grad()

            up_scale_embed = None
            index_s = 0
            up_mask_clean = None
            pred_1 = None
            pred_embed_1 = None
            for scale_value in args.scale_list:
                if scale_value == args.scale_list[0]:
                    pred_embed = model_list[index_s](downsample_torch(x, sample_rate=scale_value, device=device))

                    # update features
                    if epoch > args.warmup_epoch:
                        pred_embed = args.moment_alpha * pred_embed + (1. - args.moment_alpha) * up_outputs_list[
                                                                                                     index_s][
                                                                                                 _start:_end]

                    up_scale_embed = pred_embed
                    pred = classifier_list[index_s](pred_embed)
                    step_select_loss = loss_sample_select_list[index_s](pred, y)
                    pred_1 = pred
                    pred_embed_1 = pred_embed

                    if epoch > args.warmup_epoch:
                        if args.small_loss_criterion == 0:  ## use jsd small loss criterion
                            up_mask_clean = mask_clean[index_s][_start:_end]
                        else:
                            target_pred_label = torch.argmax(pred.data, axis=1).cpu().numpy()
                            up_mask_clean = get_clean_class_loss_ind(loss_all=step_select_loss.cpu().detach().numpy(),
                                                                     remember_rate=1 - rate_schedule[epoch],
                                                                     class_num=args.num_classes,
                                                                     predict_label=target_pred_label)
                    else:
                        if args.small_loss_criterion == 0:  ## use jsd small loss criterion
                            up_mask_clean = np.ones((_end - _start))
                        else:
                            up_mask_clean = get_clean_loss_ind(loss_all=step_select_loss.cpu().detach().numpy(),
                                                               remember_rate=1 - rate_schedule[epoch])

                    up_outputs_list[index_s][_start:_end] = pred_embed.data.clone()
                else:
                    pred_embed = model_list[index_s](downsample_torch(x, sample_rate=scale_value, device=device),
                                                     scale_x=up_scale_embed, is_head=True)
                    if epoch > args.warmup_epoch:
                        pred_embed = args.moment_alpha * pred_embed + (1. - args.moment_alpha) * up_outputs_list[
                                                                                                     index_s][
                                                                                                 _start:_end]

                    up_scale_embed = pred_embed
                    pred_s = classifier_list[index_s](pred_embed)

                    end_knn_label_y, end_clean_mask_y = None, None

                    if epoch > args.epoch_correct_start:
                        end_knn_label, end_clean_mask = construct_lp_graph(data_embed=pred_embed,
                                                                           y_label=y,
                                                                           mask_label=up_mask_clean,
                                                                           device=device, topk=args.knn_num,
                                                                           num_real_class=args.num_classes)

                        end_knn_label_y = end_knn_label
                        end_clean_mask_y = end_clean_mask

                    if end_clean_mask_y is not None:
                        step_loss = loss_list[index_s](pred_s[end_clean_mask_y == 1],
                                                       end_knn_label_y[end_clean_mask_y == 1])
                    else:
                        step_loss = loss_list[index_s](pred_s[up_mask_clean == 1], y[up_mask_clean == 1])

                    step_loss.backward(retain_graph=True)

                    if epoch > args.warmup_epoch:
                        if args.small_loss_criterion == 0:  ## use jsd small loss criterion
                            up_mask_clean = mask_clean[index_s][_start:_end]
                        else:
                            target_pred_label = torch.argmax(pred.data, axis=1).cpu().numpy()
                            up_mask_clean = get_clean_class_loss_ind(loss_all=step_select_loss.cpu().detach().numpy(),
                                                                     remember_rate=1 - rate_schedule[epoch],
                                                                     class_num=args.num_classes,
                                                                     predict_label=target_pred_label)
                    else:
                        if args.small_loss_criterion == 0:  ## use jsd small loss criterion
                            up_mask_clean = np.ones((_end - _start))
                        else:
                            up_mask_clean = get_clean_loss_ind(loss_all=step_select_loss.cpu().detach().numpy(),
                                                               remember_rate=1 - rate_schedule[epoch])

                    up_outputs_list[index_s][_start:_end] = pred_embed.data.clone()

                    if index_s == (args.scale_nums - 1):
                        epoch_train_loss += step_loss.item()

                index_s = index_s + 1

            end_knn_label_y1, end_clean_mask_y1 = None, None
            if epoch > args.epoch_correct_start:
                end_knn_label, end_clean_mask = construct_lp_graph(data_embed=pred_embed_1,
                                                                   y_label=y,
                                                                   mask_label=up_mask_clean,
                                                                   device=device, topk=args.knn_num,
                                                                   num_real_class=args.num_classes)

                end_knn_label_y1 = end_knn_label
                end_clean_mask_y1 = end_clean_mask

            if end_knn_label_y1 is not None:
                step_loss_1 = loss_list[0](pred_1[end_clean_mask_y1 == 1], end_knn_label_y1[end_clean_mask_y1 == 1])
            else:
                step_loss_1 = loss_list[0](pred_1[up_mask_clean == 1], y[up_mask_clean == 1])

            step_loss_1.backward(retain_graph=True)
            epoch_train_loss += step_loss_1.item()

            for s in range(args.scale_nums):
                optimizer_list[s].step()

            num_iterations = num_iterations + 1

        epoch_train_loss = epoch_train_loss / train_dataset.shape[0]

        if (epoch + 5) >= args.epoch:
            for s in range(args.scale_nums):
                model_list[s].eval()
                classifier_list[s].eval()

            test_loss, test_accuracy = evaluate_scale_flow_acc(test_loader, model_list, classifier_list,
                                                                           loss_list,
                                                                           args.scale_list, device)

            test_end_five_accuracies.append(test_accuracy)

        if epoch % 50 == 0:
            print("epoch : {}, train loss: {}".format(epoch, epoch_train_loss))

    train_time = time.time() - t

    print("Training end: test_acc = ", round(np.mean(test_end_five_accuracies), 4),
          "traning time (seconds) = ", round(train_time, 4))

    print('Done!')
