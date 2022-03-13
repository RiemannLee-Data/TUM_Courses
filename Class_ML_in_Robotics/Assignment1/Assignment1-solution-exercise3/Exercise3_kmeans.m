function [  ] = Exercise3_kmeans( gesture_l, gesture_o, gesture_x, init_cluster_l,init_cluster_o,init_cluster_x, k)

%% restructure data
data_set = cell(1,3);
data_set{1} = reshape(gesture_l,600,3);
data_set{2}= reshape(gesture_o,600,3);
data_set{3}= reshape(gesture_x,600,3);

center = cell(1,3);
center{1} = init_cluster_l;
center{2} = init_cluster_o;
center{3} = init_cluster_x;

%% run k_means
n = size(data_set{1},1);
final_decrement = 1e-6;
labels = zeros(3,n);
old_distortion = 0;

%compute k-means for all gestures
for i = 1:3
    decrement = inf;
    
    while( decrement > final_decrement)
        % E-step:
        % compute the euclidean distance between each point and each center
        % and then assign to cluster with closest center
        min_dist = ones(3,n)*inf;
        for N = 1:n
            for K = 1:k
                eu_dist = norm ( data_set{i}(N,:) - center{i}(K,:));
                if eu_dist < min_dist(i,N)
                    min_dist(i,N) = eu_dist;
                    labels(i,N) = K;
                end
            end
        end
        % M-step:
        % update cluster centers
        for K =1:k
            center{i}(K,:) = mean(data_set{i}(labels(i,:)'==K,:));
        end
        % compute total distortion
        distortion = sum(min_dist(i,:));
        decrement = abs(distortion - old_distortion);
        old_distortion = distortion;
    end
end

plot_results(data_set, labels)

end

