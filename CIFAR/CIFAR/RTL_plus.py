import numpy as np
import sklearn
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import normalize
import random
import torch
import os

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

from sklearn.preprocessing import StandardScaler
class RTL_OOD(object):

    def __init__(self, reduce="none", d=5, norm = 'l2', ood_score = None):
        self.ood_score = ood_score
        self.initial_embed(reduce, d)
        self.initial_norm(norm)
        self.gamma_in = self.gamma_residual
        print(self.__dict__)
        self.scaler = StandardScaler()

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
                                #  normalize=True,
                                warm_start=True, selection='cyclic')
        X_hat = self.scaler.fit_transform(X_hat)
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

rtl_ood = RTL_OOD()

import pandas as pd
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    import os
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader
    import utils.score_calculation as lib

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=10, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str, default='cifar10_wrn_pretrained', help='Method name.')

# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.') # 1 or 0 
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
# EG and benchmark details
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', default='MSP', type=str, help='score options: MSP|energy')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin|MSP')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
#RTL_reranking
parser.add_argument('--alpha', type=float, default=1e-5, help='regularization for Lasso')
parser.add_argument('--percent', type=str, default = "0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0", help='percent of samples used')
parser.add_argument('--save_name', type=str, default = "result.csv", help='percent of samples used')
parser.add_argument('--exp_num', type=int, default = 0, help='ordinal of experiments')

args = parser.parse_args()
args.percent = [float(i) for i in args.percent.split(",")]
args.save_dirs = ["./ood_results/method_{}/score_{}/exp_num_{}/alpha_{}/percent_{}".format(args.method_name, args.score, args.exp_num, args.alpha, i) for i in args.percent]
for i in args.save_dirs:
    os.makedirs(i, exist_ok = True)
args.csv_paths = [i + "/" + args.save_name for i in args.save_dirs]
output_csv = {i:open(j, 'w') for i,j in zip(args.percent, args.csv_paths)}
[output_csv[i].write('dataset,FPR,AUROC,AUPR\n') for i in output_csv]
[output_csv[i].flush() for i in output_csv]
print(args)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if 'cifar10_' in args.method_name:
    test_data = dset.CIFAR10('./data/cifar10py/', train=False, transform=test_transform, download = True)
    num_classes = 10
else:
    test_data = dset.CIFAR100('./data/cifar100py' , train=False, transform=test_transform, download = True)
    num_classes = 100


test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=True,
                                          num_workers=args.prefetch, pin_memory=False)

# Create model
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

start_epoch = 0

# Restore model
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        if 'pretrained' in args.method_name:
            subdir = 'pretrained'
        model_name = os.path.join(os.path.join(args.load, subdir), args.method_name + '_epoch_' + str(i) + '.pt') 
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume "+model_name


net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().detach().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _feature = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break
            data = data.cuda() #

            features, output = net.forward_with_feature(data)
            _feature.append(to_np(features))
            smax = to_np(F.softmax(output/ args.T, dim=1)) 
            if args.score == 'energy':
                _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
            elif args.score == 'MSP':
                _score.append(-np.max(smax, axis=1))
            elif args.score == "MaxLogit":
                _score.append(-np.max(to_np(output) , axis=1))
            elif args.score == "xent":
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                raise RuntimeError
            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)
                _right_score.append(_score[-1][right_indices])
                _wrong_score.append(_score[-1][wrong_indices])

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy(), concat(_feature).copy()
    else:
        return concat(_score)[:ood_num_examples].copy(), concat(_feature)[:ood_num_examples].copy()
if args.score == 'Odin':
    # separated because no grad is not applied
    in_score, right_score, wrong_score, in_feature = lib.get_ood_scores_odin_rtl(test_loader, net, args.test_bs, ood_num_examples, args.T, args.noise, in_dist=True)
elif args.score == 'M':
    from torch.autograd import Variable
    _, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)


    if 'cifar10_' in args.method_name:
        train_data = dset.CIFAR10('./data/cifar10py/', train=True, transform=test_transform)
    else:
        train_data = dset.CIFAR100('./data/cifar100py/', train=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False, 
                                          num_workers=args.prefetch, pin_memory=False)
    num_batches = ood_num_examples // args.test_bs

    temp_x = torch.rand(2,3,32,32)
    temp_x = Variable(temp_x)
    temp_x = temp_x.cuda()
    temp_list = net.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance', count)
    sample_mean, precision = lib.sample_estimator(net, num_classes, feature_list, train_loader) 
    in_score,M_dist_1 = lib.get_GEM_Mahalanobis_score(net, test_loader, num_classes, sample_mean, precision, count-1, args.noise, num_batches, in_dist=True)
    print(in_score[-3:], in_score[-103:-100])
elif args.score == 'GEM':
    from torch.autograd import Variable
    _, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)


    if 'cifar10_' in args.method_name:
        train_data = dset.CIFAR10('./data/cifar10py/', train=True, transform=test_transform)
    else:
        train_data = dset.CIFAR100('./data/cifar100py/', train=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False, 
                                          num_workers=args.prefetch, pin_memory=False)
    num_batches = ood_num_examples // args.test_bs

    temp_x = torch.rand(2,3,32,32)
    temp_x = Variable(temp_x)
    temp_x = temp_x.cuda() #
    temp_list = net.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance', count)
    sample_mean, precision = lib.sample_estimator(net, num_classes, feature_list, train_loader) 
    in_score,M_dist_1 = lib.get_GEM_Mahalanobis_score(net, test_loader, num_classes, sample_mean, precision, count-1, args.noise, num_batches, in_dist=True, GEM=1)
    print(in_score[-3:], in_score[-103:-100])
else:
    in_score, right_score, wrong_score, in_feature = get_ood_scores(test_loader, in_dist=True)
num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=args.method_name)

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = {i:[] for i in args.percent}, {i:[] for i in args.percent}, {i:[] for i in args.percent}
M_list=[]

def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = {i:[] for i in args.percent}, {i:[] for i in args.percent}, {i:[] for i in args.percent}

    for _ in range(num_to_avg):
        if args.score == 'Odin':
            out_score, out_feature = lib.get_ood_scores_odin_rtl(ood_loader, net, args.test_bs, ood_num_examples, args.T, args.noise)
        elif args.score == 'M':
            out_score,M_dist_2 = lib.get_GEM_Mahalanobis_score(net, ood_loader, num_classes, sample_mean, precision, count-1, args.noise, num_batches)
        elif args.score == 'GEM':
            out_score,M_dist_2 = lib.get_GEM_Mahalanobis_score(net, ood_loader, num_classes, sample_mean, precision, count-1, args.noise, num_batches,GEM=1)
        else:
            out_score, out_feature = get_ood_scores(ood_loader)

        num_in = len(in_score)
        print("num_in",num_in)
        in_and_out_score = np.concatenate((in_score, out_score)).reshape(-1, 1)
        in_and_out_feature = np.concatenate((in_feature, out_feature))
        score_cal = rtl_ood.reranking_list(in_and_out_feature, in_and_out_score, args.percent, args.alpha)
        print("successful calculate linear calibration!")
        
        for a_percent in args.percent:
            a_score_cal = score_cal[a_percent]
            in_score_changed, out_score = a_score_cal[:num_in], a_score_cal[num_in:]
            if args.out_as_pos: # OE's defines out samples as positive
                measures = get_measures(out_score, in_score_changed)
            else:
                measures = get_measures(-in_score_changed, -out_score)
            aurocs[a_percent].append(measures[0]); auprs[a_percent].append(measures[1]); fprs[a_percent].append(measures[2])
    for a_percent in args.percent:
        auroc = np.mean(aurocs[a_percent]); aupr = np.mean(auprs[a_percent]); fpr = np.mean(fprs[a_percent])
        auroc_list[a_percent].append(auroc); aupr_list[a_percent].append(aupr); fpr_list[a_percent].append(fpr)

        if num_to_avg >= 5:
            print_measures_with_std(aurocs[a_percent], auprs[a_percent], fprs[a_percent], args.method_name + "_percent_{}".format(a_percent))
        else:
            print_measures(auroc, aupr, fpr, args.method_name + "_percent_{}".format(a_percent))
            
        output_csv[a_percent].write("{},{:.2f},{:.2f},{:.2f}\n".format(ood_loader.name, 100*np.mean(fprs[a_percent]), 100*np.mean(aurocs[a_percent]), 100*np.mean(auprs[a_percent])))
        output_csv[a_percent].flush()

# /////////////// Textures ///////////////
ood_data = dset.ImageFolder(root="./data/dtd/images/",
                        transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                               trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                     num_workers=0, pin_memory=False)
ood_loader.name = "Textures"
print('\n\nTexture Detection')
get_and_print_results(ood_loader)

# /////////////// SVHN /////////////// # cropped and no sampling of the test set
ood_data = svhn.SVHN(root='./data/SVHN/', split="test",
                     transform=trn.Compose(
                         [#trn.Resize(32), 
                         trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=0, pin_memory=False)
ood_loader.name = "SVHN"
print('\n\nSVHN Detection')
get_and_print_results(ood_loader)

# /////////////// Places365 ///////////////
ood_data = dset.ImageFolder(root="./data/Places365/",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=0, pin_memory=False)
ood_loader.name = "Places365"
print('\n\nPlaces365 Detection')
get_and_print_results(ood_loader)

# /////////////// LSUN-C ///////////////
ood_data = dset.ImageFolder(root="./data/LSUN/",
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=0, pin_memory=False)
ood_loader.name = "LSUN-C"
print('\n\nLSUN_C Detection')
get_and_print_results(ood_loader)

# /////////////// LSUN-R ///////////////
ood_data = dset.ImageFolder(root="./data/LSUN_resize/",
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=0, pin_memory=False)
ood_loader.name = "LSUN-R"
print('\n\nLSUN_Resize Detection')
get_and_print_results(ood_loader)

# /////////////// iSUN ///////////////
ood_data = dset.ImageFolder(root="./data/iSUN/",
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=0, pin_memory=False)
ood_loader.name = "iSUN"
print('\n\niSUN Detection')
get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Test Results!!!!!')
for a_percent in args.percent:
    print_measures(np.mean(auroc_list[a_percent]), np.mean(aupr_list[a_percent]), np.mean(fpr_list[a_percent]), method_name=args.method_name + "_percent_{}".format(a_percent))
    output_csv[a_percent].write("{},{:.2f},{:.2f},{:.2f}\n".format("mean",100*np.mean(fpr_list[a_percent]),100*np.mean(auroc_list[a_percent]),100*np.mean(aupr_list[a_percent])))
    output_csv[a_percent].flush()
    output_csv[a_percent].close()