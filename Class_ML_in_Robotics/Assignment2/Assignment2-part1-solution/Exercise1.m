close all; clear;

load ('dataGMM');

num_comp = 4;   % no of GMM components

% a) use kmeans to initialize GMM parameters

n = length(Data);
[cluster_idx, centroids] = kmeans(Data', num_comp);

% plot data
fig = figure;
dh = plot3(Data(1,:), Data(2,:), ones(size(Data)),'r.');


mu = centroids;

for k=1:num_comp
    sigma(:,:,k) = cov(Data(:,find(cluster_idx==k))');
    mixing_coeff(k) = length(find(cluster_idx==k))/n;
end

% b) EM-estimation of GMM paramters

% start EM iterations
% --------------------------
em_on = 1;
llh_old = 0;
iter = 1;

while em_on == 1

    % E-step:
    % p(w_k | x(i), theta)

    % calc denominator
    denom = zeros(n,1);
    for ii = 1:n
        for j = 1:num_comp
            denom(ii) = denom(ii) + mixing_coeff(j) * mvnpdf(Data(:,ii)',mu(j,:),sigma(:,:,j));
        end
    end    

    for k = 1:num_comp
        for ii = 1:n
            nom(k,ii) = mixing_coeff(k) * mvnpdf(Data(:,ii)',mu(k,:),sigma(:,:,k));
            p_exp(k,ii) = nom(k,ii) / denom(ii);
        end
    end

    % M-step
    n_k = sum( p_exp,2 );

    % calc mu estimate
    for k = 1:num_comp
        mu(k,:) = 1/n_k(k) * p_exp(k,:) * Data';
    end

    % calc sigma estimate
    for k = 1:num_comp
        sigma(:,:,k) = zeros(2);
        for ii = 1:n 
            sigma(:,:,k) = sigma(:,:,k) + p_exp(k,ii) * (Data(:,ii) - mu(k,:)') * (Data(:,ii) - mu(k,:)')';
        end
        sigma(:,:,k) = 1/n_k(k) * sigma(:,:,k);
    end

    % calc mixing coefficients
    for k = 1:num_comp
        mixing_coeff(k) = n_k(k)/n;
    end

    
    % check convergence
    llh = 0; % log-likelihood
    for ii = 1:n
        inner_sum = 0;
        for k = 1:num_comp;
            inner_sum = inner_sum + mixing_coeff(k) * mvnpdf(Data(:,ii)',mu(k,:),sigma(:,:,k));
        end
        llh = llh + log(inner_sum);
    end
    llh_diff = llh - llh_old;
    llh_old = llh;

    if llh_diff < 0.01;
        em_on = 0;
    end

    table(iter, llh_diff, llh)
    iter = iter +1;

% ------------------
% end of EM
end

% vizualization
num_grid_elem = 100;
[X1,X2] = meshgrid(linspace(-0.1,0.1,num_grid_elem)', linspace(-0.1,0.1,num_grid_elem)');
X = [X1(:) X2(:)];

hold on;
for k=1:num_comp
    p = mvnpdf(X, mu(k,:), sigma(:,:,k));
    gh = surfc(X1,X2,reshape(p,num_grid_elem,num_grid_elem), 'EdgeColor','none');
    alpha(0.5);
    max(p)
end
xlabel('x');
ylabel('y');
zlabel('probability');
legend('Data');

saveas(fig, 'em_result.pdf');
