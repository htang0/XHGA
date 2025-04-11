import os
import sys
import argparse
import time
import math
import copy

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from util import AverageMeter, adjust_learning_rate, accuracy, save_model
from model import FeatureExtractor, LinearClassifierAttn
import data_pre as data

from awl import AutomaticWeightedLoss  # 多任务权值

awl = AutomaticWeightedLoss(2)  # we have 2 losses


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch_size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="num of workers to use",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of training epochs"
    )
    parser.add_argument(
        "--iterative_epochs",
        type=int,
        default=5,
        help="number of iterative training epochs",
    )

    # optimization
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default="350,400,450",
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
        default="WEAR",
        choices=["WEAR", "UTD-MHAD", "sign", "num", "abc", "hand"],
        help="dataset",
    )
    parser.add_argument(
        "--num_train_single_class",
        type=int,
        default=10,  # [600,500,400,300,200,100]*2    这里修改了2改成5
        help="num_train_basic",
    )
    parser.add_argument(
        "--num_test_single_class",
        type=int,
        default=80,  # [600,500,400,300,200,100]
        help="num_test_basic",
    )
    parser.add_argument(
        "--label_rate",
        type=int,
        default=20,  # [600,500,400,300,200,100]
        help="label_rate",
    )

    # other setting
    parser.add_argument("--cosine", action="store_true", help="using cosine annealing")
    parser.add_argument(
        "--warm", action="store_true", help="warm-up for large batch training"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to pre-trained model",
    )
    parser.add_argument(
        "--trial", type=int, default="5", help="id for recording multiple runs"
    )
    opt = parser.parse_args()

    if opt.dataset == "sign":
        opt.num_class = 6
    elif opt.dataset == "num":
        opt.num_class = 6
    elif opt.dataset == "abc":
        opt.num_class = 6
    elif opt.dataset == "hand":
        opt.num_class = 6
    elif opt.dataset == "UTD-MHAD":
        opt.num_class = 8
    elif opt.dataset == "WEAR":
        opt.num_class = 18

    # set the path according to the environment
    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_path = "./save/{}_models".format(opt.dataset)
    opt.tb_path = "./save/{}_tensorboard".format(opt.dataset)

    opt.model_name = "label_{}_lr_{}_decay_{}_bsz_{}_data_{}".format(
        opt.label_rate,
        opt.learning_rate,
        opt.weight_decay,
        opt.batch_size,
        opt.num_train_single_class,
    )

    if opt.cosine:
        opt.model_name = "{}_cosine".format(opt.model_name)

    # warm-up for large-batch training,
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


def parse_option_1():
    opt = copy.deepcopy(parse_option())
    if opt.dataset == "sign":
        opt.num_class = 8
    elif opt.dataset == "num":
        opt.num_class = 7
    elif opt.dataset == "abc":
        opt.num_class = 10
    elif opt.dataset == "hand":
        opt.num_class = 9
    elif opt.dataset == "UTD-MHAD":
        opt.num_class = 27
    elif opt.dataset == "WEAR":
        opt.num_class = 18

    return opt


def get_loader_3rd(opt):
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
        _, data_train, data_val = data.load_utd_mhad(
            opt.label_rate, objective="subject"
        )
        _, data_train1, data_val1 = data.load_utd_mhad(
            opt.label_rate, objective="action"
        )

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                *to_tensor(data_train), *to_tensor(data_train1)
            ),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*to_tensor(data_val), *to_tensor(data_val1)),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    elif opt.dataset == "WEAR":
        _, data_train, data_val = data.load_wear(opt.label_rate, objective="subject")
        _, data_train1, data_val1 = data.load_wear(opt.label_rate, objective="action")

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                *to_tensor_wear(data_train), *to_tensor_wear(data_train1)
            ),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                *to_tensor_wear(data_val), *to_tensor_wear(data_val1)
            ),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    raise ValueError(f"Unsupported dataset: {opt.dataset}")


def set_loader(opt, opt_1):
    if opt.dataset != "ours":
        return get_loader_3rd(opt)

    # load data (already normalized)
    # num_of_train = (opt.num_train_basic * opt.label_rate / 5 * np.ones(opt.num_class)).astype(int)
    # num_of_test = (opt.num_test_basic * np.ones(opt.num_class)).astype(int)
    num_of_train = (opt.num_train_single_class * np.ones(opt.num_class)).astype(int)
    num_of_test = (opt.num_test_single_class * np.ones(opt.num_class)).astype(int)

    # load labeled train and test data
    print("train labeled data:")
    x_train_label_1, x_train_label_2, y_train = data.load_data(
        opt.num_class, num_of_train, 1, opt.label_rate, opt.dataset, "s"
    )
    print("test data:")
    x_test_1, x_test_2, y_test = data.load_data(
        opt.num_class, num_of_test, 2, opt.label_rate, opt.dataset, "s"
    )

    ##下面是_1的
    num_of_train_1 = (opt_1.num_train_single_class * np.ones(opt_1.num_class)).astype(
        int
    )
    num_of_test_1 = (opt_1.num_test_single_class * np.ones(opt_1.num_class)).astype(int)

    # load labeled train and test data
    print("train labeled data_1:")
    x_train_label_1_1, x_train_label_2_1, y_train_1 = data.load_data_1(
        opt_1.num_class, num_of_train_1, 1, opt_1.label_rate, opt_1.dataset, "a"
    )
    print("test data_1:")
    x_test_1_1, x_test_2_1, y_test_1 = data.load_data_1(
        opt_1.num_class, num_of_test_1, 2, opt_1.label_rate, opt_1.dataset, "a"
    )
    ##上面是_1的

    train_dataset = data.DualMultimodalDataset(
        x_train_label_1,
        x_train_label_2,
        y_train,
        x_train_label_1_1,
        x_train_label_2_1,
        y_train_1,
    )
    test_dataset = data.DualMultimodalDataset(
        x_test_1, x_test_2, y_test, x_test_1_1, x_test_2_1, y_test_1
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    return train_loader, val_loader


def set_model(opt, opt_1):
    model = FeatureExtractor(1, opt.dataset)
    classifier = LinearClassifierAttn(opt.num_class, opt.dataset)
    classifier_1 = LinearClassifierAttn(opt_1.num_class, opt_1.dataset)
    criterion = torch.nn.CrossEntropyLoss()

    ## load pretrained feature encoders 加载预先训练的特征编码器
    ckpt = torch.load(opt.ckpt, weights_only=True)
    state_dict = ckpt["model"]

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        classifier = classifier.cuda()
        classifier_1 = classifier_1.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    model.load_state_dict(state_dict)

    # freeze the MLP in pretrained feature encoders 冻结预训练特征编码器中的多层感知机
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = False

    return model, classifier, classifier_1, criterion


def train(
    train_loader,
    model,
    classifier,
    classifier_1,
    criterion,
    optimizer,
    epoch,
    opt,
    opt_1,
):
    """one epoch training"""

    if (int(epoch / opt.iterative_epochs) % 2) == 0:
        model.train()
        classifier.train()
        classifier_1.train()
    else:
        model.eval()
        classifier.train()
        classifier_1.train()
    # if (int(epoch / opt.iterative_epochs) % 2) == 0:
    #     model.train()
    #     # classifier.eval()
    #     classifier.train()
    #     for k, v in classifier.named_parameters():
    #         v.requires_grad = False  # 固定参数
    #         for name, module in classifier.named_modules():
    #             if isinstance(module, nn.BatchNorm2d):
    #                 module.training = False
    #             elif isinstance(module, nn.BatchNorm3d):
    #                 module.training = False
    #             elif isinstance(module, nn.BatchNorm1d):
    #                 module.training = False
    #
    # else:
    #     model.eval()
    #     classifier.train()

    model.eval()
    classifier.train()
    classifier_1.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    losses_1 = AverageMeter()
    top1_1 = AverageMeter()

    end = time.time()
    for idx, (
        input_data1,
        input_data2,
        labels,
        input_data1_1,
        input_data2_1,
        labels_1,
    ) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_data1 = input_data1.cuda().float()
            input_data2 = input_data2.cuda().float()
            labels = labels.cuda()
            input_data1_1 = input_data1_1.cuda().float()
            input_data2_1 = input_data2_1.cuda().float()
            labels_1 = labels_1.cuda()
        bsz = labels.shape[0]
        bsz_1 = labels_1.shape[0]

        # compute loss
        feature1, feature2 = model.encoder(input_data1, input_data2)
        output = classifier(feature1, feature2)
        loss = criterion(output, labels)

        feature1_1, feature2_1 = model.encoder(input_data1_1, input_data2_1)
        output_1 = classifier_1(feature1_1, feature2_1)
        loss_1 = criterion(output_1, labels_1)

        # update metric
        losses.update(loss.item(), bsz)
        acc, _ = accuracy(output, labels, topk=(1, 5))
        top1.update(acc[0], bsz)

        losses_1.update(loss_1.item(), bsz_1)
        acc_1, _ = accuracy(output_1, labels_1, topk=(1, 5))
        top1_1.update(acc_1[0], bsz_1)

        total_loss = awl(loss, loss_1)
        optimizer.zero_grad()
        total_loss.backward()
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
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                )
            )
            sys.stdout.flush()

        if (idx + 1) % opt_1.print_freq == 0:
            print(
                "Train_1: [{0}][{1}/{2}]\t"
                "BT_1 {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT_1 {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss_1 {loss.val:.3f} ({loss.avg:.3f})\t"
                "Acc_1@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses_1,
                    top1=top1_1,
                )
            )
            sys.stdout.flush()

    return losses.avg, top1.avg, losses_1.avg, top1_1.avg


def validate(val_loader, model, classifier, classifier_1, criterion, opt, opt_1):
    """validation"""
    model.eval()
    classifier.eval()
    classifier_1.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    losses_1 = AverageMeter()
    top1_1 = AverageMeter()

    confusion = np.zeros((opt.num_class, opt.num_class))
    label_list = []
    pred_list = []
    confusion_1 = np.zeros((opt_1.num_class, opt_1.num_class))
    label_list_1 = []
    pred_list_1 = []

    with torch.no_grad():
        end = time.time()
        for idx, (
            input_data1,
            input_data2,
            labels,
            input_data1_1,
            input_data2_1,
            labels_1,
        ) in enumerate(
            val_loader
        ):  # 修改

            if torch.cuda.is_available():
                # device = torch.device('cuda:1')
                # input_data1 = input_data1.to(device)
                # input_data2 = input_data2.to(device)
                # labels = labels.to(device)
                input_data1 = input_data1.cuda().float()
                input_data2 = input_data2.cuda().float()
                labels = labels.cuda()
                input_data1_1 = input_data1_1.cuda().float()
                input_data2_1 = input_data2_1.cuda().float()
                labels_1 = labels_1.cuda()
            bsz = labels.shape[0]
            bsz_1 = labels_1.shape[0]

            # forward
            feature1, feature2 = model.encoder(input_data1, input_data2)
            output = classifier(feature1, feature2)
            loss = criterion(output, labels)

            feature1_1, feature2_1 = model.encoder(input_data1_1, input_data2_1)
            output_1 = classifier_1(feature1_1, feature2_1)
            loss_1 = criterion(output_1, labels_1)

            # calculate and store confusion matrix
            label_list.extend(labels.cpu().numpy())
            pred_list.extend(output.max(1)[1].cpu().numpy())

            rows = labels.cpu().numpy()
            cols = output.max(1)[1].cpu().numpy()

            for label_index in range(labels.shape[0]):
                confusion[rows[label_index], cols[label_index]] += 1

            label_list_1.extend(labels_1.cpu().numpy())
            pred_list_1.extend(output_1.max(1)[1].cpu().numpy())

            rows_1 = labels_1.cpu().numpy()
            cols_1 = output_1.max(1)[1].cpu().numpy()

            for label_index_1 in range(labels_1.shape[0]):
                confusion_1[rows_1[label_index_1], cols_1[label_index_1]] += 1

            # update metric
            losses.update(loss.item(), bsz)
            acc, _ = accuracy(output, labels, topk=(1, 5))
            top1.update(acc[0], bsz)

            losses_1.update(loss_1.item(), bsz_1)
            acc_1, _ = accuracy(output_1, labels_1, topk=(1, 5))
            top1_1.update(acc_1[0], bsz_1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        idx,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )

            if idx % opt_1.print_freq == 0:
                print(
                    "Test_1: [{0}/{1}]\t"
                    "Time_1 {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss_1 {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc_1@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        idx,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses_1,
                        top1=top1_1,
                    )
                )

    F1score_test = f1_score(label_list, pred_list, average="macro")
    F1score_test_1 = f1_score(label_list_1, pred_list_1, average="macro")

    print(
        " * Acc@1 {top1.avg:.3f}\t"
        "F1-score {F1score_test:.3f}\t".format(top1=top1, F1score_test=F1score_test)
    )
    print(
        " * Acc_1@1 {top1.avg:.3f}\t"
        "F1-score_1 {F1score_test:.3f}\t".format(
            top1=top1_1, F1score_test=F1score_test_1
        )
    )

    return (
        losses.avg,
        top1.avg,
        confusion,
        F1score_test,
        losses_1.avg,
        top1_1.avg,
        confusion_1,
        F1score_test_1,
    )


def main():
    opt = parse_option()
    opt_1 = parse_option_1()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    result_record = np.zeros((opt.trial, 3))
    result_record_1 = np.zeros((opt_1.trial, 3))

    for trial_id in range(opt.trial):

        # build data loader
        train_loader, val_loader = set_loader(opt, opt_1)

        # build model and criterion
        model, classifier, classifier_1, criterion = set_model(opt, opt_1)

        optimizer = optim.SGD(
            [  # 自适应权重
                {"params": model.parameters(), "lr": 1e-4},  # 0
                {"params": classifier.parameters(), "lr": opt.learning_rate},
                {"params": classifier_1.parameters(), "lr": opt_1.learning_rate},
                {"params": awl.parameters(), "weight_decay": 0, "lr": 1e-3},
            ],
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
        )

        record_acc = np.zeros(opt.epochs)
        record_acc_1 = np.zeros(opt_1.epochs)
        best_acc = 0
        best_acc_1 = 0
        best_val_accuracy = 0.0

        writer = SummaryWriter(opt.tb_folder + "/trial_{}".format(trial_id))

        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc, loss_1, acc_1 = train(
                train_loader,
                model,
                classifier,
                classifier_1,
                criterion,
                optimizer,
                epoch,
                opt,
                opt_1,
            )

            time2 = time.time()
            print(
                "Train epoch {}, total time {:.2f}, accuracy:{:.2f}".format(
                    epoch, time2 - time1, acc
                )
            )
            print(
                "Train epoch_1 {}, total time_1 {:.2f}, accuracy_1:{:.2f}".format(
                    epoch, time2 - time1, acc_1
                )
            )
            # eval for one epoch
            (
                loss,
                val_acc,
                confusion,
                val_F1score,
                loss_1,
                val_acc_1,
                confusion_1,
                val_F1score_1,
            ) = validate(
                val_loader, model, classifier, classifier_1, criterion, opt, opt_1
            )
            if val_acc > best_acc:
                best_acc = val_acc
                save_file = os.path.join(opt.save_folder, "best_model.pth")
                save_model(classifier, optimizer, opt, opt.epochs, save_file)
            if val_acc_1 > best_acc_1:
                best_acc_1 = val_acc_1
                save_file = os.path.join(opt.save_folder, "best_model1.pth")
                save_model(classifier_1, optimizer, opt_1, opt_1.epochs, save_file)

            writer.add_scalar("val_loss", loss, epoch)
            writer.add_scalar("val_acc", val_acc, epoch)
            writer.add_scalar("val_F1score", val_F1score, epoch)

            record_acc[epoch - 1] = val_acc
            record_acc_1[epoch - 1] = val_acc_1

        result_record[trial_id, 0] = best_acc
        result_record[trial_id, 1] = val_acc
        result_record[trial_id, 2] = val_F1score

        result_record_1[trial_id, 0] = best_acc_1
        result_record_1[trial_id, 1] = val_acc_1
        result_record_1[trial_id, 2] = val_F1score_1

        print("best accuracy: {:.2f}".format(best_acc))
        print("last accuracy: {:.3f}".format(val_acc))
        print("final F1:{:.3f}".format(val_F1score))
        print("confusion_result_labelrate_{:,}:".format(opt.label_rate))

        print("best accuracy_1: {:.2f}".format(best_acc_1))
        print("last accuracy_1: {:.3f}".format(val_acc_1))
        print("final F1_1:{:.3f}".format(val_F1score_1))
        print("confusion_1_result_labelrate_{:,}:".format(opt_1.label_rate))

        os.makedirs("./record-{}".format(opt.dataset), exist_ok=True)
        np.savetxt(
            "./record-{}/confusion_result_labelrate_{:,}_epoch{}.txt".format(
                opt.dataset,
                opt.label_rate,
                opt.iterative_epochs,
                trial_id,
            ),
            confusion,
        )
        np.savetxt(
            "./record-{}/confusion_record_acc_labelrate_{:,}_epoch_{}.txt".format(
                opt.dataset,
                opt.label_rate,
                opt.iterative_epochs,
                trial_id,
            ),
            record_acc,
        )

    print("mean best accuracy:", np.mean(result_record[:, 0]))
    print("std best accuracy:", np.std(result_record[:, 0]))
    print("mean val accuracy:", np.mean(result_record[:, 1]))
    print("std val accuracy:", np.std(result_record[:, 1]))
    print("mean val F1score:", np.mean(result_record[:, 2]))
    print("std val F1score:", np.std(result_record[:, 2]))

    print("mean best accuracy_1:", np.mean(result_record_1[:, 0]))
    print("std best accuracy_1:", np.std(result_record_1[:, 0]))
    print("mean val accuracy_1:", np.mean(result_record_1[:, 1]))
    print("std val accuracy_1:", np.std(result_record_1[:, 1]))
    print("mean val F1score_1:", np.mean(result_record_1[:, 2]))
    print("std val F1score_1:", np.std(result_record_1[:, 2]))


if __name__ == "__main__":
    main()
