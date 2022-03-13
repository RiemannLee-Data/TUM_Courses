clear all;
close all;
clc;

%% Decide number of K for k-fold cross validation.
k = 5;

%% read the given data matrix
data = load('Data.mat');
num_data = size(data.Input, 2);

%% Calculate error with varying p and different folds
max_p = 6;
esti_param = cell(max_p, k);
pos_error = zeros(1, max_p);
ori_error = zeros(1, max_p);

for p = 1:max_p
    curr_pos_error = zeros(1, k);
    curr_ori_error = zeros(1, k);
    for idx = 1:k
        % get split of data for k-fold validation
        [te_in, te_out, tr_in, tr_out] = split_data(data.Input, data.Output, k, idx);
        
        % conduct linear regression and get parameter
        curr_param = regression(tr_in, tr_out, p, p);

        % compute error with test data
        te_XYT = transform_input(te_in, p);
        te_error = te_XYT*curr_param - te_out;
        curr_pos_error(idx) = mean((te_error(:, 1).^2 + te_error(:, 2).^2).^0.5);
        curr_ori_error(idx) = mean(te_error(:, 3).^2.^0.5);
    end
    pos_error(p) = mean(curr_pos_error);
    ori_error(p) = mean(curr_ori_error);
end

%% choose optimal p1 and p2.
[~, p1] = min(pos_error);
[~, p2] = min(ori_error);

%% save the final parameter
par = regression(data.Input', data.Output', p1, p2);
save('params', 'par');

%% Simulate the robot
Simulate_robot(0,0.05)
Simulate_robot(1,0)
Simulate_robot(1,0.05)
Simulate_robot(-1,-0.05)

%%
function [XYT] = transform_input(In, P)
    n = size(In, 1);
    In = [In, In(:, 1).*In(:, 2)];
    XYT = ones(n, 1+3*P);
    for pp = 1:P
        XYT(:, 2+3*(pp-1):pp*3+1) = In.^pp;
    end
end

%%
function [param] = regression(In, Out, p1, p2)
    if p1 == p2
        XYT = transform_input(In, p1);
        param = inv(XYT'*XYT)*(XYT')*Out;
    else
        XYT_1 = transform_input(In, p1);
        XYT_2 = transform_input(In, p2);
        param_1 = inv(XYT_1'*XYT_1)*(XYT_1')*Out(:, 1);
        param_2 = inv(XYT_1'*XYT_1)*(XYT_1')*Out(:, 2);
        param_3 = inv(XYT_2'*XYT_2)*(XYT_2')*Out(:, 3);
        
        param = {param_1, param_2, param_3};
    end   
end

%%
function [te_in, te_out, tr_in, tr_out] = split_data(In, Out, k, idx)
    In = In';
    Out = Out';
    n = size(Out, 1);
    test_bool = boolean(zeros(1, n));
    if k > 1
        test_idx = [1+(idx-1)*round(n/k), idx*round(n/k)];
        test_bool(test_idx(1):test_idx(2)) = 1;
        te_in = In(test_bool, :);
        te_out = Out(test_bool, :);
        tr_in = In(~test_bool, :);
        tr_out = Out(~test_bool, :);

    elseif k == 1
        te_in = [];
        te_out = [];
        tr_in = In;
        tr_out = Out;
    end
end
