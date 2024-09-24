import random
import numpy as np
import torch
import random
import os
import ipdb

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(1)

from utils.test_utils import get_measures
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--score",type=str, choices = ["MSP", "energy", "gradnorm", "xent", "ODIN"])
parser.add_argument("--alpha",type=float, default = 1e-5)
parser.add_argument("--reduce_method",type=str, default = "none")
parser.add_argument("--reduce_dim",type=int, default = 128)
args = parser.parse_args()
#
args.percent = ["no","linear"]
args.save_dirs = "./ood_results/reduced_method_{}_dim_{}_alpha_{}_wo_slice_linear".format(args.reduce_method, args.reduce_dim, args.alpha)
os.makedirs(args.save_dirs, exist_ok = True)
args.save_files = ["./ood_results/reduced_method_{}_dim_{}_alpha_{}_wo_slice_linear/score_{}_percent_{}.csv".format(args.reduce_method, args.reduce_dim, args.alpha, args.score , i) for i in args.percent]
output_csv = {i:open(j, 'w') for i,j in zip(args.percent, args.save_files)}
[output_csv[i].write('dataset,FPR,AUROC,AUPR_in,AUPR_out\n') for i in output_csv]
[output_csv[i].flush() for i in output_csv]

import pickle
with open("./save_feature/imagenet_val_features","rb") as f:
    in_feature = pickle.load(f)

with open("./save_feature/imagenet_val_{}".format(args.score),"rb") as f:
    in_score = pickle.load(f)

import sklearn
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import normalize

class RTL_OOD(object):

    def __init__(self, reduce="none", d=5, norm = 'l2', ood_score = None):
        self.ood_score = ood_score
        self.initial_embed(reduce, d)
        self.initial_norm(norm)
        self.gamma_in = self.gamma_residual
        print(self.__dict__)

    def linear(self, X, y, *args, **kwargs):
        return {"no":y, "linear": self.linear_reranking(X, y, range(len(y)))}

    def linear_reranking(self, X, y = None, index = None):
        self.linear_reg = LinearRegression()
        X_in = X[index]
        y_in = y[index]
        self.linear_reg.fit(X_in, y_in)
        return self.linear_reg.predict(X)
    
    def reranking(self, X, y = None, percent = 0.6, alpha = 1e-5):
        if y is None:
            y = self.ood_score(X).reshape(-1,1)
        gammas = self.compute_gamma(X, y, alpha)
        index_in = np.argsort(gammas)[:int(len(X) * percent)]
        return self.linear_reranking(X, y, index_in)
    
    def reranking_list(self, X, y = None, percent = None, alpha = 1e-5):
        if y is None:
            y = self.ood_score(X).reshape(-1,1)
        if not isinstance(percent, list):
            pass
        gammas = self.compute_gamma(X, y, alpha)
        results = {}
        for a_percent in percent:
            index_in = np.argsort(gammas)[:int(len(X) * a_percent)]
            self.linear_reranking(X, y, index_in)
            results[a_percent] = self.linear_reg.predict(X)
        return results
    
    def compute_gamma(self, X, y = None, alpha = 1e-5):
        if y is None:
            y = self.ood_score(X).reshape(-1,1)
        normed_X = self.norm(X)
        X = self.embed(normed_X)
        H = np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T)
        X_hat = np.eye(H.shape[0]) - H
        y_hat = np.dot(X_hat, y)
        return self.gamma_in(X_hat, y_hat, alpha)

 
    def gamma_residual(self, X_hat, y_hat, alpha):
        self.elasticnet = ElasticNet(alpha=alpha, l1_ratio=1.0, fit_intercept=True,
                                 normalize=True, warm_start=True, selection='cyclic')
        self.elasticnet.fit(X_hat, y_hat)
        print("self.elasticnet.coef_:", self.elasticnet.coef_.shape)
        if self.elasticnet.coef_.ndim == 2:
            #[n_target, n_feature]
            coefs = self.elasticnet.coef_.T
            #[n_feature, n_target] = [n_sample, n_target]
            coefs = np.sum(np.abs(coefs), axis=1)
            return coefs 
        elif self.elasticnet.coef_.ndim == 1:
            #[n_feature] = [n_sample]
            coefs = np.abs(self.elasticnet.coef_)
            return coefs
        else:
            raise RuntimeError

    def initial_embed(self, reduce, d):
        reduce = reduce.lower()  
        assert reduce in ['isomap', 'ltsa', 'mds', 'lle', 'se', 'pca', 'none']
        if reduce == 'isomap':
            from sklearn.manifold import Isomap
            embed = Isomap(n_components=d)
        elif reduce == 'ltsa':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d,
                                           n_neighbors=5, method='ltsa')
        elif reduce == 'mds':
            from sklearn.manifold import MDS
            embed = MDS(n_components=d, metric=False)
        elif reduce == 'lle':
            from sklearn.manifold import LocallyLinearEmbedding
            embed = LocallyLinearEmbedding(n_components=d, n_neighbors=5,eigen_solver='dense')
        elif reduce == 'se':
            from sklearn.manifold import SpectralEmbedding
            embed = SpectralEmbedding(n_components=d)
        elif reduce == 'pca':
            from sklearn.decomposition import PCA
            embed = PCA(n_components=d)
        if reduce == 'none':
            self.embed = lambda x: x
        else:
            self.embed = lambda x: embed.fit_transform(x)

    def initial_norm(self, norm):
        norm = norm.lower()
        assert norm in ['l2', 'none']
        if norm == 'l2':
            self.norm = lambda x: normalize(x)
        else:
            self.norm = lambda x: x

rtl_ood = RTL_OOD(reduce=args.reduce_method, d=args.reduce_dim)

auroc_list, aupr_in_list, aupr_out_list, fpr_list = {i:[] for i in args.percent}, {i:[] for i in args.percent}, {i:[] for i in args.percent}, {i:[] for i in args.percent}

def get_dataset_results(ood_feature, ood_score, ood_name):
    in_and_out_score = np.concatenate((in_score, ood_score)).reshape(-1, 1)
    in_and_out_feature = np.concatenate((in_feature, ood_feature))
    final_scores = rtl_ood.linear_reranking(in_and_out_feature, in_and_out_score, range(len(in_and_out_feature)))
    
    for a_percent in args.percent:
        if a_percent == "no":
            scores = in_and_out_score
        else:
            scores = final_scores
        in_score_changed, out_score = scores[:len(in_feature)].reshape((-1, 1)), scores[len(in_feature):].reshape((-1, 1))

        auroc, aupr_in, aupr_out, fpr95 = get_measures(in_score_changed, out_score)
        auroc_list[a_percent].append(auroc); aupr_in_list[a_percent].append(aupr_in);
        aupr_out_list[a_percent].append(aupr_out); fpr_list[a_percent].append(fpr95);

        print('============Results for {} {}============'.format(ood_name, a_percent))
        print('AUROC: {}'.format(auroc))
        print('AUPR (In): {}'.format(aupr_in))
        print('AUPR (Out): {}'.format(aupr_out))
        print('FPR95: {}'.format(fpr95))
        output_csv[a_percent].write("{},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(ood_name, 100*fpr95, 100*auroc, 100*aupr_in, 100*aupr_out))
        output_csv[a_percent].flush()

with open("./save_feature/iNaturalist_features","rb") as f:
    out_feature = pickle.load(f)

    with open("./save_feature/iNaturalist_{}".format(args.score),"rb") as f:
        out_score = pickle.load(f)

    get_dataset_results(out_feature, out_score, "iNaturalist")

with open("./save_feature/Places_features","rb") as f:
    out_feature = pickle.load(f)

    with open("./save_feature/Places_{}".format(args.score),"rb") as f:
        out_score = pickle.load(f)

    get_dataset_results(out_feature, out_score, "Places")

with open("./save_feature/SUN_features","rb") as f:
    out_feature = pickle.load(f)

    with open("./save_feature/SUN_{}".format(args.score),"rb") as f:
        out_score = pickle.load(f)

    get_dataset_results(out_feature, out_score, "SUN")

with open("./save_feature/Textures_features","rb") as f:
    out_feature = pickle.load(f)

    with open("./save_feature/Textures_{}".format(args.score),"rb") as f:
        out_score = pickle.load(f)

    get_dataset_results(out_feature, out_score, "Textures")

for a_percent in args.percent:
    print('============Results for mean {}============'.format(a_percent))
    print('AUROC: {}'.format(np.mean(auroc_list[a_percent])))
    print('AUPR (In): {}'.format(np.mean(aupr_in_list[a_percent])))
    print('AUPR (Out): {}'.format(np.mean(aupr_out_list[a_percent])))
    print('FPR95: {}'.format(np.mean(fpr_list[a_percent])))
    output_csv[a_percent].write("{},{:.2f},{:.2f},{:.2f},{:.2f}\n".format("mean",100*np.mean(fpr_list[a_percent]),100*np.mean(auroc_list[a_percent]),100*np.mean(aupr_in_list[a_percent]),100*np.mean(aupr_out_list[a_percent])))
    output_csv[a_percent].flush()
    output_csv[a_percent].close()