import time
import numpy as np
from scipy.linalg import norm
from sklearn.metrics.pairwise import rbf_kernel
import cvxopt as cvx
from src.NDOCSVM_fun import gen_adjmat, my_inv
import matplotlib.pyplot as plt
from pprint import pprint
from src.metrics import metrics
from src.load_dataset_ADBench import *

class MyPms:
    def __init__(self, nodes, neighbors):
        self.nodes = nodes 
        self.neighbors = neighbors
        self.gamma = 10
        self.nu = 0.001
        self.ill_thres = 0.01
        self.max_repeat = 1
        self.iterations = 150

def pnd_ocsvm(dataset_path, nodes = 40, neighbors = 2):
    pms = MyPms(nodes, neighbors)
    x_train_total, y_train_total, x_test_total,y_test_total = fraud(dataset_path)
    x_test_total = x_test_total.T
    
    split_train = np.array_split(x_train_total, pms.nodes, axis=0)
    for rank in range(pms.nodes):
        split_train[rank] = split_train[rank].T

    print(pms.neighbors)
    print(pms.nodes)

    rho1 = 5
    rho2 = 0.5
   
    for t in range(pms.max_repeat):
        # 1.1 Calculate adjacent matrix. 
        adj_mat = gen_adjmat(pms.nodes, pms.neighbors)

        N_j_nei = [[] for i in range(pms.nodes)]
        for i in range(pms.nodes):
            N_j_nei[i] = np.where(adj_mat[:, i] == 1)
            N_j_nei_list = N_j_nei[i][0].tolist()
            N_j_nei[i] =[i] + N_j_nei_list

        e_j=[[] for i in range(pms.nodes)]
        for rank in range(pms.nodes):
            e_j[rank] = np.zeros((pms.nodes, 1))
            e_j[rank][rank] = 1

        Q_j=[[] for i in range(pms.nodes)]
        for rank in range(pms.nodes):
            Q_j[rank]=np.zeros((pms.nodes, len(N_j_nei[rank])))
            for i in range(len(N_j_nei[rank])):
                nei_tmp=N_j_nei[rank][i]
                Q_j[rank][:,i]=e_j[nei_tmp].flatten()

        local_n = [[] for i in range(pms.nodes)]
        for i in range(pms.nodes):
            local_n[i] = len(split_train[i].T) 

            
    # -----distributed OCSVM-----
        # 2.1 Calculate and boardcast kernel matrices.
        t0 = time.time()
        inv_K_mat_j = [[] for i in range(pms.nodes)]
        K_mat_j = [[[[] for i in range(pms.nodes)] for j in range(pms.nodes)] for k in range(pms.nodes)]
        for rank in range(pms.nodes):
            for nei_iter_1 in range(len(N_j_nei[rank])):
                nei_tmp1 = N_j_nei[rank][nei_iter_1]
                for nei_iter_2 in range(len(N_j_nei[rank])):
                    nei_tmp2 = N_j_nei[rank][nei_iter_2]
                    data_1 = split_train[nei_tmp1]
                    data_2 = split_train[nei_tmp2]
                    tmp_kernel = rbf_kernel(data_1.T, data_2.T, pms.gamma)
                    K_mat_j[rank][nei_tmp1][nei_tmp2]=tmp_kernel
            inv_K_mat_j[rank] = my_inv(K_mat_j[rank][rank][rank])
        print('Kernel matrix:', time.time() - t0)

        #2.2 Initialize 
        phi_z = [[[] for i in range(pms.nodes)] for j in range(pms.nodes)]
        for rank in range(pms.nodes):
            for nei_iter_1 in range(len(N_j_nei[rank])):
                nei_tmp1 = N_j_nei[rank][nei_iter_1]
                phi_z[nei_tmp1][rank] = np.zeros((local_n[nei_tmp1],1))
               
        b = np.zeros((pms.nodes, 1))
        phi_z_old = [[[] for i in range(pms.nodes)] for j in range(pms.nodes)]
        
        phi_alpha = [[] for i in range(pms.nodes)]
        for rank in range(pms.nodes):
            phi_alpha[rank]=np.zeros((local_n[rank], len(N_j_nei[rank])))
        beta = np.zeros((pms.nodes,1))

        lambda_j = [[] for i in range(pms.nodes)]
   
        L_value = np.zeros((pms.nodes, 1))
        L_value_old = np.zeros((pms.nodes, 1))

        rho = np.zeros((pms.nodes,1))
        H_hat = np.zeros((pms.nodes, 1))
        A = np.zeros((pms.nodes, 1))
            
        update_flag = np.ones((pms.nodes,1))
        update_thres = 1e-10#1e-3
        stop_flag = 1e-4
        
        #3. ADMM START
        for ADMM_iter in range(pms.iterations):
            print(ADMM_iter)
            b_flag = 0
            phi_z_flag = 0
            for rank in range(pms.nodes):
                rho[rank] = rho1
                A[rank] = 1 + rho[rank] * len(N_j_nei[rank])
                H_hat[rank] = 1 / (A[rank]-1)
                
            #compute lambda_j
            t0 = time.time()
            for rank in range(pms.nodes):
                if update_flag[rank] > update_thres:
                    my_P = K_mat_j[rank][rank][rank] / A[rank] \
                        + np.ones((local_n[rank], local_n[rank])) / (rho2 * len(N_j_nei[rank]))
                
                    phi_z_sum = np.zeros((local_n[rank], 1))
                    for i in range(len(N_j_nei[rank])):
                        nei_tmp = N_j_nei[rank][i]
                        phi_z_sum = phi_z_sum + phi_z[rank][nei_tmp]

                    tmp1 = rho[rank] *  phi_z_sum / A[rank] 
                    tmp2 = phi_alpha[rank]@np.ones((len(N_j_nei[rank]),1)) / A[rank]

                    b_sum=0
                    for i in range(len(N_j_nei[rank])):
                        tmp_iter = N_j_nei[rank][i]
                        b_sum = b_sum + b[tmp_iter]

                    tmp3 = (b_sum / len(N_j_nei[rank])) * np.ones((local_n[rank], 1))
                    tmp4 = np.ones((local_n[rank], 1)) / (rho2 * len(N_j_nei[rank]))

                    tmp5 = (beta[rank] / rho2) * np.ones((local_n[rank], 1))
                    my_q = tmp1 - tmp2 - tmp3 - tmp4 + tmp5

                    I = np.identity(local_n[rank])
                    zero = np.zeros(local_n[rank])
                    jci = np.full(local_n[rank], 1 / (pms.nu * local_n[rank] ))

                    q = cvx.matrix(my_q)
                    P = cvx.matrix(my_P)
                    G = cvx.matrix(np.concatenate((-I, I)))
                    h = cvx.matrix(np.concatenate((zero, jci)))
                    
                    lambda_j[rank] = np.array(cvx.solvers.qp(P, q, G, h)['x'])

            print('compute lambda:', time.time() - t0)
            
            
            #update b    
            t0 = time.time()
            for rank in range(pms.nodes):
                if update_flag[rank] > update_thres:
                    b_tmp = 0
                    for i in range(len(N_j_nei[rank])):  
                        tmp_iter = N_j_nei[rank][i]
                        b_tmp=b_tmp+b[tmp_iter]
                
                    b[rank] = (rho2 * b_tmp + 1 - len(N_j_nei[rank]) * beta[rank] - lambda_j[rank].T@np.ones((local_n[rank], 1))) \
                        / (rho2 * len(N_j_nei[rank]))
                   
            print('update b:', time.time() - t0)
            # calculate convergence of b
            # delta_b = 0
            # for rank in range(pms.nodes):
            #     for nei_iter_1 in range(len(N_j_nei[rank])):
            #             nei_tmp1 = N_j_nei[rank][nei_iter_1]
            #             delta_b = delta_b + norm(b_old[rank] - b[rank])
            # if delta_b < pms.nodes * update_thres:
            #     b_flag = 1
            # b_old = b

                    
            #update phi_z
            t0=time.time()
            for rank in range(pms.nodes):
                if update_flag[rank] > update_thres:
                    for nei_iter_1 in range(len(N_j_nei[rank])):
                        nei_tmp1 = N_j_nei[rank][nei_iter_1]
                        phi_z[nei_tmp1][rank] = np.zeros((local_n[nei_tmp1], 1))
                        for nei_iter_2 in range(len(N_j_nei[rank])):
                            nei_tmp2 = N_j_nei[rank][nei_iter_2]                           
                            tt1 = inv_K_mat_j[nei_tmp2] @ phi_alpha[nei_tmp2]
                            tt2=rho2*(lambda_j[nei_tmp2]@np.ones((1,len(N_j_nei[nei_tmp2]))))
                            tmp = K_mat_j[rank][nei_tmp1][nei_tmp2] @ (tt1+tt2)@ Q_j[nei_tmp2].T @ e_j[rank] * H_hat[rank]
                            
                            phi_z[nei_tmp1][rank] = phi_z[nei_tmp1][rank] + tmp
                   
            print('update phi_z:', time.time() - t0)
            # calculate convergence of phi_z
            # delta_phi_z = 0
            # for rank in range(pms.nodes):
            #     for nei_iter_1 in range(len(N_j_nei[rank])):
            #             nei_tmp1 = N_j_nei[rank][nei_iter_1]
            #             delta_phi_z = delta_phi_z + norm(phi_z_old[rank][nei_tmp1] - phi_z[rank][nei_tmp1])
            # if delta_phi_z < pms.nodes * update_thres:
            #     phi_z_flag = 1
            # phi_z_old = phi_z
        
            
            #update phi_alpha
            t0 = time.time()
            for rank in range(pms.nodes):
                sum_K_phi_z = np.zeros((local_n[rank], 1))
                phi_z_Q_j=np.zeros((local_n[rank],len(N_j_nei[rank])))
                for nei_iter_1 in range(len(N_j_nei[rank])):
                    nei_tmp1 = N_j_nei[rank][nei_iter_1]
                    sum_K_phi_z = sum_K_phi_z + phi_z[rank][nei_tmp1]
                    phi_z_Q_j[:,nei_iter_1]=phi_z[rank][nei_tmp1].flatten()
                tmp_1 = (rho[rank] * sum_K_phi_z + K_mat_j[rank][rank][rank] @ lambda_j[rank] \
                    - phi_alpha[rank] @ np.ones((len(N_j_nei[rank]), 1))) @ np.ones((1, len(N_j_nei[rank]))) \
                        / A[rank]
                phi_alpha[rank] = phi_alpha[rank] + rho[rank] * (tmp_1 - phi_z_Q_j)
               
            print('update phi_alpha:', time.time() - t0)         

            # update beta
            t0=time.time()
            for rank in range(pms.nodes):
                sum_b=0
                for nei_iter_1 in range(len(N_j_nei[rank])):
                    nei_tmp = N_j_nei[rank][nei_iter_1]
                    sum_b = b[rank] - b[nei_tmp] + sum_b
                beta[rank] = beta[rank] + rho2 * sum_b
            print('update beta', time.time() - t0)

            # calculate Lagrangian function 
            delta_L = 0
            stop_criter = 3e-5*pms.nodes#1e-3
            for rank in range(pms.nodes):
                w1 = -lambda_j[rank].T @ K_mat_j[rank][rank][rank] @ lambda_j[rank] / (2 * A[rank])
                phi_z_sum = np.zeros((local_n[rank], 1))
                for i in range(len(N_j_nei[rank])):
                    nei_tmp = N_j_nei[rank][i]
                    phi_z_sum = phi_z_sum + phi_z[rank][nei_tmp]
                w2 = -(rho[rank] * phi_z_sum).T @ lambda_j[rank] / A[rank]
                w3 = (phi_alpha[rank] @ np.ones((len(N_j_nei[rank]), 1))).T @ lambda_j[rank] / A[rank]
                w_related = w1 + w2 + w3

                phiz_related = -(phi_alpha[rank].T @ inv_K_mat_j[rank] @ phi_z_sum).trace() + 0.5 * rho[rank] * (phi_z_sum.T @ inv_K_mat_j[rank] @ phi_z_sum)
                 
                b1 = -b[rank] * (1 + lambda_j[rank].T @ np.ones((local_n[rank], 1)))
                b2 = 0
                for nei_iter_1 in range(len(N_j_nei[rank])):
                    nei_tmp = N_j_nei[rank][nei_iter_1]
                    sum_b = b[rank] - b[nei_tmp]
                    b2 += sum_b * beta[rank] + 0.5 * rho[rank] * np.linalg.norm(sum_b)
                b_related = b1 + b2

                L_value[rank] = (w_related + phiz_related + b_related)
                if ADMM_iter > 1:
                    delta_L = delta_L + norm(L_value[rank] - L_value_old[rank])
                    print(delta_L)
                L_value_old[rank] = L_value[rank]
                
            # if delta_L!=0 and delta_L < stop_criter:
            #     break

        predict = [[] for i in range(pms.nodes)]
        y_scores = [[] for i in range(pms.nodes)]
        tpr_total = 0
        fpr_total = 0
        f1_list = []
        auc_list = []

        # test  
        for rank in range(pms.nodes):
            data_1 = split_train[rank]
            data_2 = x_test_total
            K_mat_test = rbf_kernel(data_2.T,data_1.T, pms.gamma)
            g = K_mat_test @ inv_K_mat_j[rank] @ phi_z[rank][rank] \
                - b[rank] * np.ones((len(x_test_total.T),1))

            y_scores[rank]=g.reshape(-1)
            predict[rank] = np.sign(y_scores[rank])


            output_metrics = metrics(y_test_total, predict[rank], y_scores[rank])

            tpr_total +=  output_metrics['TPR']
            fpr_total +=  output_metrics['FPR']
            f1_list.append( output_metrics['F1'])
            auc_list.append( output_metrics['AUROC'])
            pprint(metrics(y_test_total, predict[rank], y_scores[rank]))

        tpr_average = tpr_total / pms.nodes
        fpr_average = fpr_total / pms.nodes
        f1_list_mean = np.mean(f1_list)
        f1_list_std = np.std(f1_list)
        auc_list_mean = np.mean(auc_list)
        auc_list_std = np.std(auc_list)

        print("tpr_average:", tpr_average)
        print("fpr_average:", fpr_average)
        print("f1_list_mean:", f1_list_mean)
        print("f1_list_std:", f1_list_std)
        print("auc_list_mean:", auc_list_mean)
        print("auc_list_std:", auc_list_std)
