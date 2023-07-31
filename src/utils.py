import pickle
import numpy as np
import random
import torch
import pywt
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score, precision_recall_curve, \
    roc_curve, auc, f1_score, average_precision_score



def row_normalize(a_matrix):
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1) + 1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix


def standardization(data):
    mu = np.mean(data, axis=1, keepdims=True)
    sigma = np.std(data, axis=1, keepdims=True)
    data = (data - mu) / (sigma)
    data[np.isnan(data) | np.isinf(data)] = 0.0
    return data


def dse_normalize(cuda, drug_se, D_n=1020, S_n=5599):
    se_drug = drug_se.T
    drug_se_normalize = torch.from_numpy(row_normalize(drug_se)).float()
    se_drug_normalize = torch.from_numpy(row_normalize(se_drug)).float()
    if cuda:
        drug_se_normalize = drug_se_normalize.cuda()
        se_drug_normalize = se_drug_normalize.cuda()
    return drug_se_normalize, se_drug_normalize


def gen_adj(A):
    D = torch.pow(A.sum(1), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def wavelet_encoder(seq):
    meta_drug = np.array(list(map(lambda x: int(x, 16), seq)))
    ca, cd = pywt.dwt(meta_drug, 'db1')
    drug_feature = ca / np.sum(ca)
    return drug_feature


def load_data(path="../data/", mpnn="mpnn_toxcast.npy", weave="weave_toxcast.npy", afp="afp_toxcast.npy",
              nf="nf_toxcast.npy", fpt="drugs.fpt", num=1020):
    print('Loading features...')
    fpt_feature = np.zeros((num, 128))
    drug_file = open(path + fpt, "r")
    index = 0
    for line in drug_file:
        line = line.strip()
        if line == "":
            hex_arr = np.zeros(128)
        else:
            hex_arr = wavelet_encoder(line)
        fpt_feature[index] = hex_arr
        index += 1
    fpt_feature = torch.FloatTensor(standardization(fpt_feature))
    mpnn_feature = torch.FloatTensor(standardization(np.load(path + mpnn)))
    weave_feature = torch.FloatTensor(standardization(np.load(path + weave)))
    afp_feature = torch.FloatTensor(standardization(np.load(path + afp)))
    nf_feature = torch.FloatTensor(standardization(np.load(path + nf)))
    with open(f"{path}/mols_vec.pkl", 'rb') as f:
        vec_feature = torch.FloatTensor(standardization(pickle.load(f)))
    drug_file.close()
    return fpt_feature, mpnn_feature, weave_feature, afp_feature, nf_feature, vec_feature


def get_links(path="../data/", dataset="drug_se_matrix.txt"):
    drug_se = np.loadtxt(path + dataset)
    data_set = drug_se.flatten()
    return data_set


def sample_links(data, seed, pos_count, neg_count):
    random.seed(seed)
    pos_list = []
    neg_list = []
    for data_tmp in data:
        if data_tmp[-1] == 1:
            pos_list.append(data_tmp)
        else:
            neg_list.append(data_tmp)
    pos_data = random.sample(pos_list, pos_count)
    neg_data = random.sample(neg_list, neg_count)
    return np.array(pos_data + neg_data)


def save_result(outputs, data_set, test_mask, fold, path="../result/",
                D_n=1020, S_n=5599):
    mask = torch.from_numpy(np.where(data_set.reshape(D_n, S_n) == 1, 0, 1)).cuda()
    matrix = torch.mul(torch.mul(outputs, mask), test_mask)
    result = []
    for i in range(D_n):
        for j in range(S_n):
            if matrix[i][j] != 0:
                # print(matrix[i][j])
                result.append([torch.sigmoid(matrix[i][j]).cpu().detach(), i, j])
    result.sort(key=lambda item: item[0], reverse=True)
    np.save(path + 'case_fold' + str(fold), result)


def save_all(final_outputs, test_mask, fold, path="../result/"):
    np.save(path + 'result' + str(fold), final_outputs.cpu().detach().numpy())
    np.save(path + 'mask' + str(fold), test_mask.cpu().detach().numpy())


def validation(y_pre, y, flag=False):
    prec, recall, _ = precision_recall_curve(y, y_pre)
    pr_auc = auc(recall, prec)
    mr = mrank(y, y_pre)
    if flag:
        fpr, tpr, threshold = roc_curve(y, y_pre)
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(y, y_pre)
        y_predict_class = y_pre
        y_predict_class[y_predict_class > 0.5] = 1
        y_predict_class[y_predict_class <= 0.5] = 0
        prec = precision_score(y, y_predict_class)
        recall = recall_score(y, y_predict_class)
        mcc = matthews_corrcoef(y, y_predict_class)
        f1 = f1_score(y, y_predict_class)
        return roc_auc, pr_auc, prec, recall, mcc, f1, ap, mr
    return mr, pr_auc, _, _, _, _, _, _


def mrank(y, y_pre):
    index = np.argsort(-y_pre)
    r_label = y[index]
    r_index = np.array(np.where(r_label == 1)) + 1
    reci_sum = np.sum(1 / r_index)
    # reci_rank = np.mean(1 / r_index)
    return reci_sum


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
