from __future__ import division
from __future__ import print_function

import os
import logging
import time
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils import load_data, get_links, dse_normalize, validation, save_all
from model import DSEModel
import warnings
warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../data/', help='path to data')
parser.add_argument('--result_path', type=str, default='../result/', help='path to result')
parser.add_argument('--model_path', type=str, default='../model/', help='path to save model')
parser.add_argument('--log_dir', nargs='?', default='../log', help='Input data path.')
parser.add_argument('--D_n', type=int, default=1020, help='number of drug node')
parser.add_argument('--S_n', type=int, default=5599, help='number of side-effect node')
parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hid_dim', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--fpt_dim', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--vec_dim', type=int, default=300, help='Number of hidden units.')
parser.add_argument('--gnn_dim', type=int, default=617, help='Number of hidden units.')
parser.add_argument('--vec_len', type=int, default=100, help='Number of hidden units.')
parser.add_argument('--kge_dim', type=int, default=400, help='Number of hidden units.')
parser.add_argument('--bio_dim', type=int, default=768, help='Number of hidden units.')
parser.add_argument('--alpha', type=float, default=0.02, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=50, help='Patience')


# ----------------------------------------define log information--------------------------------------------------------

# create log information
def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".txt")
    print("All logs will be saved to %s" % logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


def get_device(args):
    args.gpu = False
    if torch.cuda.is_available() and args.cuda:
        args.gpu = True
        print(f'Training on GPU.')
    else:
        print(f'Training on CPU.')
    device = torch.device("cuda:0" if args.gpu else "cpu")
    return device


args = parser.parse_args()
# set log file
log_save_id = create_log_id(args.log_dir)
logging_config(folder=args.log_dir, name='log{:d}'.format(log_save_id), no_console=False)
logging.info(args)
device = get_device(args)
# seed
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Load data
fpt_feature, mpnn_feature, weave_feature, afp_feature, nf_feature, vec_feature = load_data()
data_set = get_links(path=args.data_path, dataset="drug_se_matrix.txt")


# if args.cuda:
fpt_feature = fpt_feature.to(device)
mpnn_feature = mpnn_feature.to(device)
weave_feature = weave_feature.to(device)
afp_feature = afp_feature.to(device)
nf_feature = nf_feature.to(device)
vec_feature = vec_feature.to(device)
loss_m = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([40]).cuda())
loss_f = torch.nn.MSELoss()


def train(model, optimizer, mask, target, train_idx, train_set):
    model.train()
    optimizer.zero_grad()
    outputs= model()
    output = torch.flatten(torch.mul(mask, outputs))
    loss_train = loss_m(output, target)
    loss_train.backward()
    optimizer.step()
    output = output[train_idx]
    noutput = torch.sigmoid(output).cpu().detach().numpy()
    metrics = validation(noutput, train_set)
    return loss_train.data.item(), metrics[0], metrics[1], outputs


def compute_test(test_set, outputs, mask, test_idx, flag=False):
    output = torch.flatten(torch.mul(mask, outputs))[test_idx]
    noutput = torch.sigmoid(output).cpu().detach().numpy()
    metrics = validation(noutput, test_set, flag)
    return metrics


kf = StratifiedKFold(n_splits=10, shuffle=True)
counter = 1
auc_arr = []
aupr_arr = []
mcc_arr = []
f1_arr = []
prec_arr = []
recall_arr = []
ap_arr = []
mr_arr = []
valid_aupr_arr = []
index = 0


for train_index, test_index in kf.split(data_set, data_set):
    train_index, valid_index = train_test_split(train_index, test_size=0.05)
    train_set = data_set[train_index]
    valid_set = data_set[valid_index]
    print("train shape:", train_set.shape, ", valid shape:", valid_set.shape)
    test_set = data_set[test_index]
    logging.info('Begin {:02d}th folder, train_size:{:02d}, train_label:{:.2f}, valid_label:{:.2f}, test_label:{:.2f}'
                 .format(counter, len(train_index), np.sum(train_set), np.sum(valid_set), np.sum(test_set)))
    train_mask = np.zeros(args.D_n * args.S_n)
    train_mask[train_index] = 1
    target = np.multiply(data_set, train_mask)
    matrix = target.reshape(args.D_n, args.S_n)

    logging.info('train_mask:{}, matrix {}'.format(np.sum(train_mask), np.sum(matrix)))

    train_mask = torch.from_numpy(train_mask.reshape(args.D_n, args.S_n)).to(device)
    target = torch.from_numpy(target).to(device)

    drug_se_train, se_drug_train = dse_normalize(device, matrix, D_n=args.D_n, S_n=args.S_n)

    test_mask = np.zeros(args.D_n * args.S_n)
    test_mask[test_index] = 1
    test_mask = torch.from_numpy(test_mask.reshape(args.D_n, args.S_n)).to(device)

    valid_mask = np.zeros(args.D_n * args.S_n)
    valid_mask[valid_index] = 1
    valid_mask = torch.from_numpy(valid_mask.reshape(args.D_n, args.S_n)).to(device)

    model = DSEModel(args, fpt_feature, mpnn_feature, weave_feature, afp_feature, nf_feature, vec_feature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    # Train model
    t_total = time.time()
    bad_counter = 0
    best_epoch = 0
    best_loss = 0
    final_outputs = []

    for epoch in range(args.epochs):
        auc, aupr, outputs = [], [], []
        loss = 0
        t = time.time()
        loss, train_mrr, train_aupr, outputs = train(model, optimizer, train_mask, target, train_index, train_set)
        valid_metrics = compute_test(valid_set, outputs, valid_mask, valid_index)
        valid_mrr, valid_aupr = valid_metrics[0], valid_metrics[1]
        test_metrics = compute_test(test_set, outputs, test_mask, test_index)
        test_mrr, test_aupr = test_metrics[0], test_metrics[1]
        logging.info('time: {:.4f}s, train_mrr: {:.4f}, train_aupr: {:.4f}, valid_mrr: {:.4f}, valid_aupr: {:.4f}, loss_train: {:.4f}'.format((time.time() - t), train_mrr, train_aupr, valid_mrr, valid_aupr, loss))
        logging.info('folder= {:02d}, Epoch: {:04d}, test_mrr: {:.4f}, test_aupr: {:.4f}, Best_epoch: {:04d}'.format(counter, (epoch+1), test_mrr, test_aupr, (best_epoch+1)))
        if valid_aupr > best_loss:
            best_loss = valid_aupr
            best_epoch = epoch
            bad_counter = 0
            final_outputs = outputs
        else:
            bad_counter += 1

        if bad_counter >= args.patience:
            break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    logging.info('Loading {:04d}th epoch'.format(best_epoch))

    # Testing
    # save_result(final_outputs, data_set, test_mask, counter)
    save_all(final_outputs, test_mask, counter)
    test_auc, test_aupr, prec, recall, mcc, f1, ap, mr = compute_test(test_set, final_outputs, test_mask, test_index, True)
    logging.info('Test set results:, folder= {:02d}, test_auc: {:.4f}, test_aupr: {:.4f}, test_mcc: {:.4f}, '
                 'test_f1: {:.4f}, test_ap: {:.4f}, test_mr: {:.4f}, test_prec: {:.4f}, test_recall: {:.4f}'.format(
        counter, test_auc, test_aupr, mcc, f1, ap, mr, prec, recall))
    valid_aupr_arr.append(best_loss)
    auc_arr.append(test_auc)
    aupr_arr.append(test_aupr)
    mcc_arr.append(mcc)
    f1_arr.append(f1)
    prec_arr.append(prec)
    recall_arr.append(recall)
    ap_arr.append(ap)
    mr_arr.append(mr)
    np.savetxt(args.result_path + 'valid_aupr_avg', [counter, np.mean(np.array(valid_aupr_arr))])
    np.savetxt(args.result_path + 'auc_avg', [counter, np.mean(np.array(auc_arr))])
    np.savetxt(args.result_path + 'aupr_avg', [counter, np.mean(np.array(aupr_arr))])
    np.savetxt(args.result_path + 'mcc_avg', [counter, np.mean(np.array(mcc_arr))])
    np.savetxt(args.result_path + 'f1_avg', [counter, np.mean(np.array(f1_arr))])
    np.savetxt(args.result_path + 'prec_avg', [counter, np.mean(np.array(prec_arr))])
    np.savetxt(args.result_path + 'recall_avg', [counter, np.mean(np.array(recall_arr))])
    np.savetxt(args.result_path + 'ap_avg', [counter, np.mean(np.array(ap_arr))])
    np.savetxt(args.result_path + 'mr_avg', [counter, np.mean(np.array(mr_arr))])
    np.savetxt(args.result_path + 'valid_aupr', np.array(valid_aupr_arr))
    np.savetxt(args.result_path + 'auc', np.array(auc_arr))
    np.savetxt(args.result_path + 'aupr', np.array(aupr_arr))
    np.savetxt(args.result_path + 'mcc', np.array(mcc_arr))
    np.savetxt(args.result_path + 'f1', np.array(f1_arr))
    np.savetxt(args.result_path + 'prec', np.array(prec_arr))
    np.savetxt(args.result_path + 'recall', np.array(recall_arr))
    np.savetxt(args.result_path + 'ap', np.array(ap_arr))
    np.savetxt(args.result_path + 'mr', np.array(mr_arr))
    counter += 1
