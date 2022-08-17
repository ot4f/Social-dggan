import torch
import torch.distributions.multivariate_normal as torchdist
import numpy as np
import math
from scipy.stats import gaussian_kde
from utils import to_theta
from sklearn.mixture import GaussianMixture
from scipy.special import erf


def ade(pred, target, num_of_objs=None):
    """
    num_peds: shape (batch,)
    """
    batch_size = len(pred)
    pred_time = pred.shape[2]
    if num_of_objs is None:
        num_of_objs = [pred.shape[1]] * batch_size
    sum_all = 0
    for s in range(batch_size):
        sum_ = 0
        node_num = num_of_objs[s]
        for i in range(int(node_num)):
            for t in range(pred_time):
                sum_ += math.sqrt((pred[s][i][t][0] - target[s][i][t][0]) ** 2 + (pred[s][i][t][1] - target[s][i][t][1]) ** 2)
        sum_all += sum_ / (pred_time*node_num)
    return sum_all/batch_size

def fde(pred, target, num_of_objs=None):
    batch_size = len(pred)
    pred_time = pred.shape[2]
    if num_of_objs is None:
        num_of_objs = [pred.shape[1]] * batch_size
    sum_all = 0
    for s in range(batch_size):
        sum_ = 0
        node_num = num_of_objs[s]
        for i in range(int(node_num)):
            for t in range(pred_time - 1, pred_time):
                sum_ += math.sqrt((pred[s][i][t][0] - target[s][i][t][0]) ** 2 + (pred[s][i][t][1] - target[s][i][t][1]) ** 2)
        sum_all += sum_/(node_num)
    return sum_all/batch_size

def ade_sstgcnn(pred, target, num_of_objs=None,mean=True, theta_=False):
    """
    num_peds: shape (batch,)
    """
    batch_size = len(pred)
    pred_time = pred.shape[2]
    if num_of_objs is None:
        num_of_objs = [pred.shape[1]] * batch_size
    sum_all = 0
    for s in range(batch_size):
        # sum_ = 0
        node_num = num_of_objs[s]
        for i in range(int(node_num)):
            sum_ = 0
            for t in range(pred_time):
                if theta_:
                    epsilon = 1e-10
                    pred_v = pred[s][i,t]
                    target_v = target[s][i,t]
                    pred_v_norm = np.sqrt((pred_v**2).sum())
                    target_v_norm = np.sqrt((target_v**2).sum())
                    if pred_v_norm > epsilon and target_v_norm > epsilon:
                        sum_ += 1.0 - np.sum(pred_v*target_v)/(pred_v_norm*target_v_norm)
                    else:
                        sum_ += (1.0/(1+np.exp(-np.abs(pred_v_norm-target_v_norm)))-0.5)*4.0
                else:
                    sum_ += math.sqrt((pred[s][i][t][0] - target[s][i][t][0]) ** 2 + (pred[s][i][t][1] - target[s][i][t][1]) ** 2)
            sum_all += sum_ / pred_time
    if mean:
        return sum_all / sum(num_of_objs)
    else:
        return sum_all

def fde_sstgcnn(pred, target, num_of_objs=None, mean=True):
    batch_size = len(pred)
    pred_time = pred.shape[2]
    if num_of_objs is None:
        num_of_objs = [pred.shape[1]] * batch_size
    sum_all = 0
    for s in range(batch_size):
        # sum_ = 0
        node_num = num_of_objs[s]
        for i in range(int(node_num)):
            sum_ = 0
            for t in range(pred_time - 1, pred_time):
                sum_ += math.sqrt((pred[s][i][t][0] - target[s][i][t][0]) ** 2 + (pred[s][i][t][1] - target[s][i][t][1]) ** 2)
            sum_all += sum_
    if mean:
        return sum_all / sum(num_of_objs)
    else:
        return sum_all

def cal_aae(pred_traj, pred_traj_gt, mode='sum', epsilon=1e-12):
    """ calculating AAE
    pred_traj: ndarray of shape (seq, batch, xy)
    ppred_traj_gt: ndarray of shape (seq, batch, xy)
    """
    # def _sigmoid(x_):
    #     x = x_.copy()
    #     _mask = x < 0
    #     x[_mask] = np.exp(x[_mask])/(1.0+np.exp(x[_mask]))
    #     x[~_mask] = 1.0/(1.0+np.exp(-x[~mask]))
    #     return x

    assert pred_traj.shape[0] == 1 and pred_traj_gt.shape[0] == 1
    pred_traj = pred_traj.squeeze()
    pred_traj_gt = pred_traj_gt.squeeze()
    
    pred_traj = pred_traj.permute(1,0,2)
    pred_traj_gt = pred_traj_gt.permute(1,0,2)
    loss = (pred_traj * pred_traj_gt).sum(axis=2)
    pred_traj_norm = torch.sqrt((pred_traj**2).sum(axis=2))
    pred_traj_gt_norm = torch.sqrt((pred_traj_gt**2).sum(axis=2))
    
    mask = (pred_traj_norm > epsilon) & (pred_traj_gt_norm > epsilon)
    loss[mask] = 1.0 - loss[mask]/(pred_traj_norm[mask]*pred_traj_gt_norm[mask])
    loss[~mask] = (torch.sigmoid(torch.abs(pred_traj_norm[~mask]-pred_traj_gt_norm[~mask])) - 0.5) * 4.0
    loss = loss.sum(dim=1)

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def aae_1(pred_traj, pred_traj_gt, epsilon=1e-12):
    """
    pred: shape (1, seq, 1, 2)
    gt: shape (1, seq, 1, 2)
    """
    loss = (pred_traj * pred_traj_gt).sum(axis=-1)
    pred_traj_norm = torch.sqrt((pred_traj**2).sum(axis=-1))
    pred_traj_gt_norm = torch.sqrt((pred_traj_gt**2).sum(axis=-1))
    
    mask = (pred_traj_norm > epsilon) & (pred_traj_gt_norm > epsilon)
    loss[mask] = 1.0 - loss[mask]/(pred_traj_norm[mask]*pred_traj_gt_norm[mask])
    loss[~mask] = (torch.sigmoid(torch.abs(pred_traj_norm[~mask]-pred_traj_gt_norm[~mask])) - 0.5) * 4.0
    loss = loss.mean()
    return loss

def ade_1(pred, target, count, theta_=False):
    assert pred[0].shape[0] == 1
    n = len(pred)
    sum_all = 0
    for s in range(n):
        pred_ = np.swapaxes(pred[s][:,:,:count[s],:],1,2)  # [batch, node, sq, feat]
        target_ = np.swapaxes(target[s][:,:,:count[s],:],1,2)
        
        pred_ = np.squeeze(pred_, 0)
        target_ = np.squeeze(target_, 0)

        N = pred_.shape[0]
        T = pred_.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                if theta_:
                    pred_theta = to_theta(pred_[i,t])
                    # target_theta = to_theta(target_[i,t])
                    # sum_ += np.abs(pred_theta-target_theta)
                    epsilon = 1e-10
                    pred_v = pred_[i,t]
                    target_v = target_[i,t]
                    pred_v_norm = np.sqrt((pred_v**2).sum())
                    target_v_norm = np.sqrt((target_v**2).sum())
                    if pred_v_norm > epsilon and target_v_norm > epsilon:
                        sum_ += 1.0 - np.sum(pred_v*target_v)/(pred_v_norm*target_v_norm)
                    else:
                        sum_ += (1.0/(1+np.exp(-np.abs(pred_v_norm-target_v_norm)))-0.5)*4.0
                else:
                    sum_ += np.sqrt((pred_[i,t,0] - target_[i,t,0])**2+(pred_[i,t,1] - target_[i,t,1])**2)
        sum_all += sum_/(N*T)
    return sum_all/n

def fde_1(pred, target, count, theta_=False):
    assert pred[0].shape[0] == 1
    n = len(pred)
    sum_all = 0
    for s in range(n):
        pred_ = np.swapaxes(pred[s][:,:,:count[s],:],1,2)  # [batch, node, sq, feat]
        target_ = np.swapaxes(target[s][:,:,:count[s],:],1,2)
        
        pred_ = np.squeeze(pred_, 0)
        target_ = np.squeeze(target_, 0)

        N = pred_.shape[0]
        T = pred_.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T-1,T):
                if theta_:
                    pred_theta = to_theta(pred_[i,t])
                    target_theta = to_theta(target_[i,t])
                    sum_ += np.abs(pred_theta-target_theta)
                else:
                    sum_ += np.sqrt((pred_[i,t,0] - target_[i,t,0])**2+(pred_[i,t,1] - target_[i,t,1])**2)
        sum_all += sum_/(N)
    return sum_all/n

def MSE(input, target):
    mse = ((input-target)**2).mean()
    return float(mse)

def EdgeWiseKL(input, target):
    mask = (input > 0) & (target > 0)
    input = input[mask]
    target = target[mask]
    kl = (target * np.log(target / input)).mean()
    return float(kl)

def EdgeWiseKL_T(input, target):
    mask = (input > 0) & (target > 0)
    input = input[mask]
    target = target[mask]
    kl = (target * torch.log(target / input)).mean()
    return float(kl)

def get_class_acc(y, pred, n_classes):
    y = y.data.cpu()
    pred = pred.data.cpu()
    y = y.reshape(-1, n_classes)
    pred = pred.reshape(-1, n_classes)
    xi, yi = torch.where(y!=0)
    a, b = torch.where(pred[xi]==pred[xi].max(dim=1)[0].unsqueeze(1))
    pred_ = [b[0]]
    for i in range(1, len(a)):
        if a[i] != a[i-1]:
            pred_.append(b[i])
    pred = torch.LongTensor(pred_)
    acc = torch.mean((pred==yi).type(torch.float))
    return acc

def bivariate_loss(V_pred,V_trgt):
    """
    V_pred: tensor, shape (1, seq, node, feat)
    """

    normx = V_trgt[...,0]- V_pred[...,0]
    normy = V_trgt[...,1]- V_pred[...,1]

    sx = torch.exp(V_pred[...,2]) #sx
    sy = torch.exp(V_pred[...,3]) #sy
    corr = torch.tanh(V_pred[...,4]) #corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho)) + 1e-10

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)
    
    return result

def kde_nll_1(predicted_trajs, gt_traj):
    """
    :param predicted_trajs: ndarray, shape [1, num_samples, seq_len, 2]
    :param gt_traj: ndarray, shape [seq_len, 2]
    """
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf
            except np.linalg.LinAlgError:
                kde_ll = np.nan
    kde_ll = kde_ll / (num_timesteps * num_batches)
    return -kde_ll

# AMD / AMV
def calc_amd_amv(gt, pred):
    total = 0
    m_collect = []
    gmm_cov_all = 0
    for i in range(pred.shape[0]):  #per time step
        for j in range(pred.shape[1]):
            #do the method of finding the best bic
            temp = pred[i, j, :, :]

            gmm = get_best_gmm2(pred[i, j, :, :])
            center = np.sum(np.multiply(gmm.means_, gmm.weights_[:,
                                                                 np.newaxis]),
                            axis=0)
            gmm_cov = 0
            for cnt in range(len(gmm.means_)):
                gmm_cov += gmm.weights_[cnt] * (
                    gmm.means_[cnt] - center)[..., None] @ np.transpose(
                        (gmm.means_[cnt] - center)[..., None])
            gmm_cov = np.sum(gmm.weights_[..., None, None] * gmm.covariances_,
                             axis=0) + gmm_cov

            dist, _ = mahalanobis_d(
                center, gt[i, j], len(gmm.weights_), gmm.covariances_,
                gmm.means_, gmm.weights_
            )  #assume it will be the true value, add parameters

            total += dist
            gmm_cov_all += gmm_cov
            m_collect.append(dist)

    gmm_cov_all = gmm_cov_all / (pred.shape[0] * pred.shape[1])
    return total / (pred.shape[0] *
                    pred.shape[1]), None, None, m_collect, np.abs(
                        np.linalg.eigvals(gmm_cov_all)).max()

def mahalanobis_d(x, y, n_clusters, ccov, cmeans, cluster_p):  #ccov
    v = np.array(x - y)
    Gnum = 0
    Gden = 0
    for i in range(0, n_clusters):
        ck = np.linalg.pinv(ccov[i])
        u = np.array(cmeans[i] - y)
        val = ck * cluster_p[i]
        b2 = 1 / (v.T @ ck @ v)
        a = b2 * v.T @ ck @ u
        Z = u.T @ ck @ u - b2 * (v.T @ ck @ u)**2
        pxk = np.sqrt(np.pi * b2 / 2) * np.exp(-Z / 2) * (erf(
            (1 - a) / np.sqrt(2 * b2)) - erf(-a / np.sqrt(2 * b2)))
        Gnum += val * pxk
        Gden += cluster_p[i] * pxk
    G = Gnum / Gden
    mdist = np.sqrt(v.T @ G @ v)
    if np.isnan(mdist):
        # print(Gnum, Gden)
        '''
        print("is nan")
        print(v)
        print("Number of clusters", n_clusters)
        print("covariances", ccov)
        '''
        return 0, 0

    # print( "Mahalanobis distance between " + str(x) + " and "+str(y) + " is "+ str(mdist) )
    return mdist, G

def get_best_gmm(X):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(
        1, 7)  ## stop based on fit/small BIC change/ earlystopping
    cv_types = ['full']
    best_gmm = GaussianMixture()
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components,
                                  covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    return best_gmm

def get_best_gmm2(X):  #early stopping gmm
    lowest_bic = np.infty
    bic = []
    cv_types = ['full']  #changed to only looking for full covariance
    best_gmm = GaussianMixture()
    for cv_type in cv_types:
        p = 1  #Decide a value
        n_comps = 1
        j = 0
        while j < p and n_comps < 5:  # if hasn't improved in p times, then stop. Do it for each cv type and take the minimum of all of them
            gmm = GaussianMixture(n_components=n_comps,
                                  covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                j = 0  #reset counter
            else:  #increment counter
                j += 1
            n_comps += 1

    bic = np.array(bic)
    return best_gmm

def kde_lossf(gt, pred):
    #(12, objects, samples, 2)
    # 12, 1600,1000,2
    kde_ll = 0
    kde_ll_f = 0
    n_u_c = 0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            temp = pred[i, j, :, :]
            n_unique = len(np.unique(temp, axis=0))
            if n_unique > 2:
                kde = gaussian_kde(pred[i, j, :, :].T)
                t = np.clip(kde.logpdf(gt[i, j, :].T), a_min=-20,
                            a_max=None)[0]
                kde_ll += t
                if i == (pred.shape[0] - 1):
                    kde_ll_f += t
            else:
                n_u_c += 1
    if n_u_c == pred.shape[0] * pred.shape[1]:
        return 0
    return -kde_ll / (pred.shape[0] * pred.shape[1])