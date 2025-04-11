import os
import sys
import argparse
import time
import math
import itertools

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.backends.cudnn as cudnn

from model import FeatureExtractor, ConFusionLoss
from util import AverageMeter, adjust_learning_rate, set_optimizer, save_model
import data_pre as data


def FeatureConstructor(f1, f2, num_positive):
    fusion_weight = np.arange(1, num_positive + 1) / 10  # (0.1, 0,2, ..., 0.9)
    # a = [random.uniform(0.1, 0.9) for _ in range(9)]

    fused_feature = []

    for fuse_id in range(num_positive):
        temp_fuse = fusion_weight[fuse_id] * f1 + (1 - fusion_weight[fuse_id]) * f2
        # temp_fuse = a[fuse_id] * f1 + (1 - a[fuse_id]) * f2
        fused_feature.append(temp_fuse)

    fused_feature = torch.stack(fused_feature, dim=1)

    return fused_feature


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=5, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")  # 32
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="num of workers to use",
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="number of training epochs"
    )

    # optimization
    parser.add_argument(
        "--learning_rate", type=float, default=1e-2, help="learning rate"
    )
    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default="100,200,300",
        help="where to decay lr, can be a list",
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.9, help="decay rate for learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")

    parser.add_argument(
        "--dataset",
        type=str,
        default="UTD-MHAD",
        choices=["UTD-MHAD", "WEAR", "ours"],
        help="dataset",
    )
    parser.add_argument("--label_rate", type=int, default=20, help="label_rate")

    parser.add_argument("--num_positive", type=int, default=9, help="num_positive")

    # temperature
    parser.add_argument(
        "--temp", type=float, default=0.07, help="temperature for loss function"
    )

    # other setting
    parser.add_argument("--cosine", action="store_true", help="using cosine annealing")
    parser.add_argument(
        "--warm", action="store_true", help="warm-up for large batch training"
    )

    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_path = "./save/{}_models".format(opt.dataset)
    opt.tb_path = "./save/{}_tensorboard".format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = "label_{}_lr_{}_decay_{}_bsz_{}_temp_{}_epoch_{}".format(
        opt.label_rate,
        opt.learning_rate,
        opt.lr_decay_rate,
        opt.batch_size,
        opt.temp,
        opt.epochs,
    )

    if opt.cosine:
        opt.model_name = "{}_cosine".format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = "{}_warm".format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate**3)
            opt.warmup_to = (
                eta_min
                + (opt.learning_rate - eta_min)
                * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs))
                / 2
            )
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_loader_3rd(opt):
    def mask_indices(data, limit_class):
        label_all = list(enumerate(data[2].tolist()))
        inertial_all = data[0]
        rgb_all = data[1]
        indices = []
        for (_, group), _ in zip(
            itertools.groupby(label_all, lambda v: v[1][0]), range(limit_class)
        ):
            for i, _ in group:
                indices.append(i)
        inertial_all = inertial_all[indices]
        rgb_all = rgb_all[indices]
        print(inertial_all.shape)
        return (inertial_all, rgb_all, data[2][indices])

    def to_tensor(data):
        return (
            torch.from_numpy(np.concatenate((data[0], data[0]), axis=2))
            .unsqueeze(1)
            .float(),
            torch.from_numpy(data[1].swapaxes(2, 4).swapaxes(1, 2)).float(),
            torch.from_numpy(data[2]).long(),
        )

    def to_tensor_wear(data):
        return (
            torch.from_numpy(data[0]).unsqueeze(1).float(),
            torch.from_numpy(data[1].swapaxes(2, 4).swapaxes(1, 2)).float(),
            torch.from_numpy(data[2]).long(),
        )

    if opt.dataset == "UTD-MHAD":
        data_all, _, _ = data.load_utd_mhad(opt.label_rate)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*to_tensor(data_all)),
            # drop 20% of the unlabelld data
            # torch.utils.data.TensorDataset(*to_tensor(mask_indices(data_all, 20))),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader
    elif opt.dataset == "WEAR":
        data_all, _, _ = data.load_wear(opt.label_rate)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*to_tensor_wear(data_all)),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader
    else:
        raise ValueError(f"Unsupported dataset: {opt.dataset}")


def set_loader(opt):
    if opt.dataset != "ours":
        return get_loader_3rd(opt)

    # load labeled train and test data
    x_train_1, x_train_2 = data.load_unlabel_data(opt.label_rate)
    print(x_train_1.shape)
    print(x_train_2.shape)

    train_dataset = data.MultimodalUnlabeledDataset(x_train_1, x_train_2)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )

    return train_loader


def set_model(opt):
    model = FeatureExtractor(1, opt.dataset)
    criterion = ConFusionLoss(temperature=opt.temp)

    if torch.cuda.is_available():
        # device = torch.device('cuda:1')
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        # model = model.to(device)
        # criterion = criterion.to(device)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (input_data1, input_data2, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            # device = torch.device('cuda:1')
            # input_data1 = input_data1.to(device)
            # input_data2 = input_data2.to(device)
            # input_data3 = input_data3.to(device)
            input_data1 = input_data1.cuda().float()
            input_data2 = input_data2.cuda().float()
        bsz = input_data1.shape[0]

        # compute loss
        feature1, feature2 = model(input_data1, input_data2)
        features = FeatureConstructor(feature1, feature2, opt.num_positive)
        loss = criterion(features)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    writer = SummaryWriter(opt.tb_folder)
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        # tensorboard logger
        writer.add_scalar("loss", loss, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, opt, opt.epochs, save_file)
    print("num_positive:", opt.num_positive)
    print("label_rate:", opt.label_rate)


if __name__ == "__main__":
    main()
