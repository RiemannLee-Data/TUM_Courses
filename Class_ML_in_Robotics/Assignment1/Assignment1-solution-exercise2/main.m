clear all;
close all;
clc;

%% defining maximum d
max_d = 60;

%% load images
tr_imgs = loadMNISTImages('train-images.idx3-ubyte');
tr_labels = loadMNISTLabels('train-labels.idx1-ubyte');
te_imgs = loadMNISTImages('t10k-images.idx3-ubyte');
te_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

%% make the training data zero mean.
tr_mean = mean(tr_imgs, 2);
tr_imgs_normed = tr_imgs - tr_mean;

%% calculate the cov
tr_cov = cov(tr_imgs_normed');

%% calculate eigenvalues/eigenvectors and sort them in order.
[vec, val] = eig(tr_cov);
[~, ind] = sort(diag(val), 'descend');
vec = vec(:, ind);

%% Try calculate error when using d:1~max_d principal components.
error = zeros(1, max_d);
total_pred_cls = zeros(size(te_labels, 1), max_d);
te_imgs_normed = te_imgs - tr_mean;
for d = 1:max_d
    likelihood = zeros(size(te_labels, 1), 10);
    curr_basis = vec(:, 1:d);
    tr_img_proj = curr_basis' * tr_imgs_normed;
    te_img_proj = curr_basis' * te_imgs_normed;
    for cls = 0:9
        mu = mean(tr_img_proj(:, find(tr_labels==cls)), 2)';
        sigma = cov(tr_img_proj(:, find(tr_labels==cls))');
        likelihood(:, cls+1) = mvnpdf(te_img_proj', mu, sigma);
    end    
    [~, curr_pred_cls] = max(likelihood');
    curr_pred_cls = (curr_pred_cls-1)';
    total_pred_cls(:, d) = curr_pred_cls;
    error(d) = mean(curr_pred_cls ~= te_labels) * 100;
end

%% find which d value is optimal
[min_err, opt_d] = min(error);
opt_pred = total_pred_cls(:, opt_d);

%% Draw the confusion map
confusion_mat = confusionmat(te_labels, opt_pred);
confusionchart(confusion_mat);
%% new 2020 matlab does not support helperdisplayconfusionmatrix?