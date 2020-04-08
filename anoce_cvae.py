''''
Main function for traininng DAG-GNN
'''

from __future__ import division
from __future__ import print_function

import time
import pickle
import os
import random

# import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import math
from allfun import *

from multiprocessing import Pool
import multiprocessing
n_cores = multiprocessing.cpu_count()
from numpy.random import randn
from random import seed as rseed
from numpy.random import seed as npseed
import os
os.environ["OMP_NUM_THREADS"] = "1"


def anoce_simu(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    #  # configurations
    data_sample_size = n = 38 #the number of samples of data
    data_variable_size = d = 32 #the number of variables in synthetic generated data
    graph_degree = degree = 5 #the number of degree in generated DAG graph
    x_dims = 1 #The number of input dimensions: default 1.
    z_dims = d #The number of latent variable dimensions: default the same as variable size.
    graph_sem_type = sem_type = 'linear-gauss' #the structure equation model (SEM) parameter type
    graph_linear_type = linear_type = 'linear' #the synthetic data type: linear -> linear SEM, nonlinear_1 -> x=Acos(x+1)+z, nonlinear_2 -> x=2sin(A(x+0.5))+A(x+0.5)+z


    # -----------training hyperparameters
    epochs = 200 #Number of epochs to train.
    batch_size = 19 #Number of samples per batch. note: should be divisible by sample size, otherwise throw an error
    optimizer = 'Adam' #'the choice of optimizer used'
    graph_threshold = 0.3,  # 0.3 is good, 0.2 is error prune, threshold for learned adjacency matrix binarization
    original_lr = 3e-3  # basline rate = 1e-3, Initial learning rate.
    encoder_hidden = 64 #Number of hidden units.
    decoder_hidden = 64 #Number of hidden units.
    temp = 0.5 #Temperature for Gumbel softmax.
    k_max_iter = 100 #the max iteration number for searching lambda, gamma, c and d
    factor = True #Factor graph model.
    encoder_dropout = 0.0 #Dropout rate (1 - keep probability).
    decoder_dropout = 0.0 #Dropout rate (1 - keep probability).
    h_tol = 1e-8 #the tolerance of error of h(A) to zero'
    g_tol = 1e-8 #the tolerance of error of g(A) to zero'
    lr_decay = 200 #After how epochs to decay LR by a factor of gamma.'
    gamma = 1.0 #LR decay factor.
    tau_A = 0. #coefficient for L-1 norm of A
    lambda_A = 0. #coefficient for DAG constraint h(A)
    c_A = 1 #coefficient for absolute value h(A).
    gamma_A = 0. #coefficient for DAG constraint g(A).
    d_A = 1 #coefficient for absolute value g(A).

    # compute constraint h(A) value
    def _h_A(A, m):
        expm_A = matrix_poly(A*A, m)
        h_A = torch.trace(expm_A) - m
        return h_A
    
    # compute constraint g(A) value
    def _g_A(A, m):
        g_A = sum(abs(A[:,0]))+sum(abs(A[m-1,:]))-abs(A[m-1,0])
        return g_A
    
    def stau(w, tau):
        w1 = prox_plus(torch.abs(w)-tau)
        return torch.sign(w)*w1


    def update_optimizer(optimizer, old_lr, c_A, d_A):
        '''related LR to c_A and d_A, whenever c_A and d_A gets big, reduce LR proportionally'''
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        estimated_lr = old_lr / (math.log10(c_A) + math.log10(d_A) + 1e-10)
        if estimated_lr > MAX_LR:
            lr = MAX_LR
        elif estimated_lr < MIN_LR:
            lr = MIN_LR
        else:
            lr = estimated_lr

        # set LR
        for parame_group in optimizer.param_groups:
            parame_group['lr'] = lr

        return optimizer, lr


    #===================================
    # training:
    #===================================

    def train(epoch, best_val_loss, ground_truth_G, lambda_A, c_A, gamma_A, d_A, optimizer, old_lr):
        t = time.time()
        nll_train = []
        kl_train = []
        mse_train = []
        shd_trian = []

        encoder.train()
        decoder.train()
        scheduler.step()

        # update optimizer
        optimizer, lr = update_optimizer(optimizer, old_lr, c_A, d_A)


        for batch_idx, (data, relations) in enumerate(train_loader):

            data, relations = Variable(data).double(), Variable(relations).double()

            # reshape data
            relations = relations.unsqueeze(2)

            optimizer.zero_grad()

            enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data, rel_rec, rel_send)  # logits is of size: [num_sims, z_dims]
            edges = logits

            dec_x, output, adj_A_tilt_decoder = decoder(data, edges, data_variable_size * x_dims, rel_rec, rel_send, origin_A, adj_A_tilt_encoder, Wa)

            if torch.sum(output != output):
                print('nan error\n')

            target = data
            preds = output
            variance = 0.

            # reconstruction accuracy loss
            loss_nll = nll_gaussian(preds, target, variance)

            # KL loss
            loss_kl = kl_gaussian_sem(logits)

            # ELBO loss:
            loss = loss_kl + loss_nll

            # add A loss
            one_adj_A = origin_A # torch.mean(adj_A_tilt_decoder, dim =0)
            sparse_loss = tau_A * torch.sum(torch.abs(one_adj_A))


            # compute h(A) and g(A)
            h_A = _h_A(origin_A, data_variable_size)
            g_A = _g_A(origin_A, data_variable_size)
            loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A +gamma_A * g_A + 0.5 * d_A * g_A * g_A+ 100. * torch.trace(origin_A*origin_A) + sparse_loss #+  0.01 * torch.sum(variance * variance)

            loss.backward()
            loss = optimizer.step()

            myA.data = stau(myA.data, tau_A*lr)

            if torch.sum(origin_A != origin_A):
                print('nan error\n')

            # compute metrics
            graph = origin_A.data.clone().numpy()

            graph[np.abs(graph) < graph_threshold] = 0
        
            
            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))

            mse_train.append(F.mse_loss(preds, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())
            shd_trian.append(shd)


        if 'graph' not in vars():
            print('error on assign')


        return np.mean(np.mean(kl_train)  + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A, optimizer, lr



    # load true G
    with open(os.path.join('', 'fakeG.pkl'), 'rb') as trueG:
        ground_truth_G = pickle.load(trueG)

    # load data
    with open(os.path.join("", 'covid19.pkl'), 'rb') as data:
        X = pickle.load(data)
    
    feat_train = torch.FloatTensor(X)
    feat_valid = torch.FloatTensor(X)
    feat_test = torch.FloatTensor(X)
    
    # reconstruct itself
    train_data = TensorDataset(feat_train, feat_train)
    valid_data = TensorDataset(feat_valid, feat_train)
    test_data = TensorDataset(feat_test, feat_train)
    
    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)


    #===================================
    # load modules
    #===================================
    # Generate off-diagonal interaction graph
    off_diag = np.ones([data_variable_size, data_variable_size]) - np.eye(data_variable_size)

    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
    rel_rec = torch.DoubleTensor(rel_rec)
    rel_send = torch.DoubleTensor(rel_send)

    # add adjacency matrix A
    num_nodes = data_variable_size
    adj_A = np.zeros((num_nodes, num_nodes))


    encoder = MLPEncoder(data_variable_size * x_dims, x_dims, encoder_hidden,
                             int(z_dims), adj_A,
                             batch_size = batch_size,
                             do_prob = encoder_dropout, factor = factor).double()
    decoder = MLPDecoder(data_variable_size * x_dims,
                             z_dims, x_dims, encoder,
                             data_variable_size = data_variable_size,
                             batch_size = batch_size,
                             n_hid=decoder_hidden,
                             do_prob=decoder_dropout).double()

    #===================================
    # set up training parameters
    #===================================
    if optimizer == 'Adam':
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=original_lr)
    elif optimizer == 'LBFGS':
        optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=original_lr)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=original_lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay,
                                    gamma=gamma)

    # Linear indices of an upper triangular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(data_variable_size)
    tril_indices = get_tril_offdiag_indices(data_variable_size)

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)

    #===================================
    # main
    #===================================

    t_total = time.time()
    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []
    # optimizer step on hyparameters
    h_A_new = torch.tensor(1.)
    g_A_new = 1
    h_A_old = np.inf
    g_A_old = np.inf
    lr=original_lr

    try:
        for step_k in range(k_max_iter):
            print("Iteration:",step_k)
            while c_A*d_A < 1e+20:
                for epoch in range(epochs):
                    old_lr=lr
                    ELBO_loss, NLL_loss, MSE_loss, graph, origin_A, optimizer, lr = train(epoch, best_ELBO_loss, ground_truth_G, lambda_A, c_A, gamma_A, d_A, optimizer, old_lr)
                    if ELBO_loss < best_ELBO_loss:
                        best_ELBO_loss = ELBO_loss
                        best_epoch = epoch
                        best_ELBO_graph = graph

                    if NLL_loss < best_NLL_loss:
                        best_NLL_loss = NLL_loss
                        best_epoch = epoch
                        best_NLL_graph = graph

                    if MSE_loss < best_MSE_loss:
                        best_MSE_loss = MSE_loss
                        best_epoch = epoch
                        best_MSE_graph = graph

                if ELBO_loss > 2 * best_ELBO_loss:
                    break

                # update parameters
                A_new = origin_A.data.clone()
                h_A_new = _h_A(A_new, data_variable_size)
                g_A_new = _g_A(A_new, data_variable_size)
                if h_A_new.item() > 0.25 * h_A_old and g_A_new > 0.25 * g_A_old:
                    c_A*=10
                    d_A*=10
                elif h_A_new.item() > 0.25 * h_A_old and g_A_new < 0.25 * g_A_old:
                    c_A*=10
                elif h_A_new.item() < 0.25 * h_A_old and g_A_new > 0.25 * g_A_old:
                    d_A*=10
                else:
                    break


                # update parameters
                # h_A, g_A, adj_A are computed in loss anyway, so no need to store
            h_A_old = h_A_new.item()
            g_A_old = g_A_new
            lambda_A += c_A * h_A_new.item()
            gamma_A += d_A * g_A_new
            
            if h_A_new.item() <= h_tol and g_A_new <= g_tol:
                break


        # test()
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
        print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
 
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
        print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
 
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
        print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)


    except KeyboardInterrupt:
 
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
        print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
 
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
        print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
        print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    matG1 = np.matrix(origin_A.data.clone().numpy())

    return matG1


seed=2333
np.random.seed(seed) #Random seed
rep=100
seeds_list=np.random.randint(1, 1000000, size=rep)

pool = Pool(n_cores)
rep_res=pool.map(anoce_simu, seeds_list)

with open('ANOCE_real.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(rep_res, filehandle)
