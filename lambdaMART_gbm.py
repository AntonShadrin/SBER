import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold

# Parameters
max_depth = 10
count_trees = 600
min_samples_leaf = 2 #8941
alpha_max = 0.1
alpha_min = 0.01
num_tree_for_restriction = 3
criterion = "mse"
sigma_ = 1.
t = 2
EPS = 1.e-20

# Load train
X_train, y_train, qid_train = load_svmlight_file(f="train.txt", query_id=True)
y_train += 1
# sort by qid
indexesSort = [x[0] for x in sorted(zip(np.array(range(y_train.shape[0])), qid_train), key=lambda z: z[1])]
X_train = X_train[indexesSort]
y_train = y_train[indexesSort]
qid_train = qid_train[indexesSort]
counts_train = np.unique(qid_train, return_counts=True)[1]

# Load test
X_test, y_test, qid_test = load_svmlight_file(f="test.txt", query_id=True)
# sort by qid
indexesSort = [x[0] for x in sorted(zip(np.array(range(qid_test.shape[0])), qid_test), key=lambda z: z[1])]
X_test = X_test[indexesSort]
# y_test = y_test[indexesSort]
qid_test = qid_test[indexesSort]
counts_test = np.unique(qid_test, return_counts=True)[1]

# calculate log2 and label_power2
max_size_qid = np.max(np.unique(qid_train, return_counts=True)[1])
logs2_inverse = np.zeros(max_size_qid + 10)
for i in range(logs2_inverse.shape[0]):
    logs2_inverse[i] = 1.0 / np.log2(i+2)
max_qid = np.max(qid_train)
label_power2 = np.zeros(max_qid+1)
for i in range(label_power2.shape[0]):
    label_power2[i] = np.power(2, i)
# calculate exponents
sigmoid_bins = 1024*1024
exponents = np.zeros(sigmoid_bins)
min_sigmoid_input_ = 50
max_sigmoid_input_ = -50
exponents.resize(sigmoid_bins)
sigmoid_table_idx_factor_ = sigmoid_bins / (max_sigmoid_input_ - min_sigmoid_input_)
for i in range(exponents.shape[0]):
  score = i / sigmoid_table_idx_factor_ + min_sigmoid_input_
  exponents[i] = 1.0 / (1.0 + np.exp(score * sigma_))


def calc_inverse_idcg(y_true):
    IDCG = 0.0
    sorted_y_true = [x[1] for x in sorted(zip(y_true, y_true), key=lambda pair: pair[0], reverse=True)]
    for i in range(len(y_true)):
        # IDCG += (y_true[i])/(np.log2(i+2))
        IDCG += (np.power(2, sorted_y_true[i]) - 1)/(np.log2(i+2))
    if IDCG != 0.0:
        return (1.0/IDCG)
    return IDCG

def ndcg(y_true, predict, inverse_IDCG):
    # if inverse_IDCG == 0.0:
    #     return 0.0
    sorted_y_true = [x[1] for x in sorted(zip(predict, y_true), key=lambda pair: pair[0], reverse=True)]
    DCG = 0.0
    for i in range(len(y_true)):
        DCG += (np.power(2, sorted_y_true[i]) - 1)/(np.log2(i+2))
    return DCG*inverse_IDCG

def create_lambdas_hessians_fast(label, score, lambdas, hessians, inverse_IDCG):
    if inverse_IDCG == 0.0:
        return

    sorted_idx = [x[1] for x in sorted(zip(score, np.arange(len(score))),
                                       key=lambda pair: pair[0], reverse=True)]
    sum_lambdas = 0.0
    for i in range(len(label)):
        high = sorted_idx[i]
        high_label = label[high]
        high_score = score[high]
        high_sum_lambda = 0.
        high_sum_hessian = 0.
        for j in range(len(label)):
            if i == j:
                continue
            low = sorted_idx[j]
            low_label = label[low]
            if high_label <= low_label:
                continue
            low_score = score[low]
            d_score = high_score - low_score
            if d_score > 50:
                d_score = 50
            if d_score < -50:
                d_score = -50
            p_lambda = exponents[int((d_score - min_sigmoid_input_) * sigmoid_table_idx_factor_)]
            p_hessian = p_lambda * (1. - p_lambda)
            d_label = label_power2[int(high_label)] - label_power2[int(low_label)]

            delta_NDCG = np.abs(d_label * (logs2_inverse[j] - logs2_inverse[i]))
            delta_NDCG = delta_NDCG * inverse_IDCG
            # delta_NDCG /= (0.01 + np.abs(d_score))

            p_lambda *= ((-sigma_) * delta_NDCG)  # sigmoid_ *
            p_hessian *= (sigma_ * sigma_ * delta_NDCG)  # sigmoid_ * sigmoid_
            high_sum_lambda += p_lambda
            high_sum_hessian += p_hessian
            lambdas[low] += p_lambda
            hessians[low] += p_hessian
            sum_lambdas -= 2 * p_lambda

        lambdas[high] -= high_sum_lambda
        hessians[high] += high_sum_hessian

def create_lambdas_hessians_fast_upgrade(label, score, lambdas, hessians, inverse_IDCG):
    if inverse_IDCG == 0.0:
        return

    query_ndcg = ndcg(label, score, inverse_IDCG)
    sorted_idx = [x[1] for x in sorted(zip(score, np.arange(len(score))),
                                       key=lambda pair: pair[0], reverse=True)]
    sum_lambdas = 0.0
    for i in range(len(label)):
        high = sorted_idx[i]
        high_label = label[high]
        high_score = score[high]
        high_sum_lambda = 0.
        high_sum_hessian = 0.
        for j in range(len(label)):
            if i == j:
                continue
            low = sorted_idx[j]
            low_label = label[low]
            if high_label <= low_label:
                continue
            low_score = score[low]
            d_score = high_score - low_score
            if d_score > 50:
                d_score = 50
            if d_score < -50:
                d_score = -50
            p_lambda = exponents[int((d_score - min_sigmoid_input_) * sigmoid_table_idx_factor_)]
            p_hessian = p_lambda * (1. - p_lambda)
            d_label = label_power2[int(high_label)] - label_power2[int(low_label)]

            delta_NDCG = d_label * (logs2_inverse[j] - logs2_inverse[i])
            delta_NDCG = delta_NDCG * inverse_IDCG
            delta_NDCG = np.abs(np.power(query_ndcg, t) - np.power(query_ndcg-delta_NDCG, t))

            # delta_NDCG /= (0.01 + np.abs(d_score))

            p_lambda *= ((-sigma_) * delta_NDCG)  # sigmoid_ *
            p_hessian *= (sigma_ * sigma_ * delta_NDCG)  # sigmoid_ * sigmoid_
            high_sum_lambda += p_lambda
            high_sum_hessian += p_hessian
            lambdas[low] += p_lambda
            hessians[low] += p_hessian
            sum_lambdas -= 2 * p_lambda

        lambdas[high] -= high_sum_lambda
        hessians[high] += high_sum_hessian

# calculate inverse IDCG for train
index = 0
inverse_idcg_train = []
for c in counts_train:
    inverse_idcg_train.append(calc_inverse_idcg(y_train[index:index + c]))
    index = index + c

# fit
start_tree = DecisionTreeRegressor(max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf, random_state=241)
start_tree.fit(X_train, y_train)

# predict
# y_predict_train = np.zeros(y_train.shape[0])
# y_predict_test = np.zeros(y_test.shape[0])
y_predict_train = start_tree.predict(X_train)
y_predict_test = start_tree.predict(X_test)
index = 0
result = np.zeros((qid_test.shape[0], 2))

# # save result
# for c in counts_test:
#     result[index:index + c] = [[int(x[1]), int(x[2])] for x in sorted(zip(y_predict_test[index:index + c],
#                                                                           qid_test[index:index + c],
#                                                                           np.arange(index+1, index + c+1)),
#                                                                       key=lambda pair: pair[0], reverse=True)]
#     index = index + c
# np.savetxt(f"res_tree_{0}.txt", result, fmt='%d', delimiter=',', header="QueryId,DocumentId")
# print(f"res_tree_{0}.txt saved!")

# # print curent ndcg
# index = 0
# schet = 0
# lst_cur_ndcg = []
# for c in counts_train:
#     lst_cur_ndcg.append(ndcg(y_train[index:index + c], y_predict_train[index:index + c],
#                              inverse_idcg_train[schet]))
#     index = index + c
#     schet += 1
# print("start_ndcg = ", np.mean(lst_cur_ndcg))

# BOOSTING
d_y = np.zeros(y_predict_train.shape[0])
d_y_test = np.zeros(y_predict_test.shape[0])
lambdas = np.zeros(y_train.shape[0])
hessians = np.zeros(y_train.shape[0])
for i in np.arange(count_trees):
    # if (i+1) % 4 == 0:
    #     max_depth += 1
    # alpha = 1.0
    alpha = alpha_max - (alpha_max - alpha_min)*(i/(count_trees-1))

    index = 0
    schet = 0
    for c in counts_train:
        lambdas[index:index + c] = 0.
        hessians[index:index + c] = 0.
        create_lambdas_hessians_fast(y_train[index:index + c], y_predict_train[index:index + c],
                                lambdas[index:index + c], hessians[index:index + c],
                                inverse_idcg_train[schet])
        index += c
        schet += 1

    # fit
    cur_tree = DecisionTreeRegressor(max_depth=4, random_state=241)
    cur_tree.fit(X_train, lambdas)
    leaf_idx = cur_tree.apply(X_train)
    map_replace = {}
    for leaf_id in np.unique(leaf_idx):
        mask = (leaf_idx == leaf_id)
        gamma = np.sum(lambdas[mask]) / (EPS + np.sum(hessians[mask]))
        map_replace[leaf_id] = gamma
        d_y[mask] = gamma
    y_predict_train += alpha * d_y

    # predict
    leaf_idx = cur_tree.apply(X_test)
    for id in np.unique(leaf_idx):
        mask = (leaf_idx == id)
        gamma = map_replace[id]
        d_y_test[mask] = gamma
    y_predict_test += alpha * d_y_test

    # save result
    index = 0
    for c in counts_test:
        result[index:index + c] = [[int(x[1]), int(x[2])] for x in sorted(zip(y_predict_test[index:index + c],
                                                                              qid_test[index:index + c],
                                                                              np.arange(index + 1, index + c + 1)),
                                                                          key=lambda pair: pair[0], reverse=True)]
        index = index + c
    np.savetxt(f"res_tree_{i+1}.txt", result, fmt='%d', delimiter=',', header="QueryId,DocumentId")
    print(f"res_tree_{i+1}.txt saved!")

    # # print curent ndcg
    # index = 0
    # schet = 0
    # lst_cur_ndcg = []
    # for c in counts_train:
    #     lst_cur_ndcg.append(ndcg(y_train[index:index + c], y_predict_train[index:index + c],
    #                              inverse_idcg_train[schet]))
    #     index = index + c
    #     schet += 1
    # print(f"res_tree_{i+1}_ndcg = ", np.mean(lst_cur_ndcg))


