function [ ] = Exercise3_nubs( gesture_l, gesture_o, gesture_x, k )

%% restructure data
data_set = cell(1,3);
data_set{1} = reshape(gesture_l,600,3);
data_set{2}= reshape(gesture_o,600,3);
data_set{3}= reshape(gesture_x,600,3);

data_mean = cell(1,3);

% split vector
v = [0.08; 0.05; 0.02]';

%% compute center
n = size(data_set{1},1)
labels = ones(3,n);

for i = 1:3
    
    data_mean{i} = mean(data_set{i},1);

    % repeat the next steps k times
    for num_classes = 1:k-1
        % compute distortion for each of the current classes
        distortion = zeros(3,k);
        for K = 1:num_classes
            for j=1:n
                % sum up over all members of that class
                if(labels(i,j)==K)
                    distortion(i,K) = distortion(i,K) + norm(data_set{i}(j,:) - data_mean{i}(K,:));
                end
            end
        end
        % find the class with largest distortion
        [~,splitclass] = max(distortion(i,:));
        
        %compute two new points relative to the center of that class
        mu_neg = data_mean{i}(splitclass,:) -v;
        mu_pos = data_mean{i}(splitclass,:) +v;
        
        %sort all members of that class according to which of the two they
        %are closer to
        class_old = [];
        class_new = [];
        
        for j=1:n
            if(labels(i,j)== splitclass)
                dist_neg =  norm(data_set{i}(j,:) -  mu_neg);
                dist_pos =  norm(data_set{i}(j,:) -  mu_pos);
                
                %sort points according to distance into old and new classes
                if(dist_pos > dist_neg)
                    labels(i,j) = num_classes+1;
                    class_new = [class_new; data_set{i}(j,:)];
                else
                    class_old = [class_old; data_set{i}(j,:)];
                end
            end
        end
        % update cluster centers
        data_mean{i}(splitclass,:) = mean(class_old,1);
        data_mean{i} = [data_mean{i};  mean(class_new,1)];  
    end
end

plot_results(data_set, labels)

end

