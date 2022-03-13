function [ output_args ] = plot_results( data_set, labels )
%PLOT_RESULTS

k = max(max(labels))
colors = jet(k)
names = ['l','o','x'];
for i = 1:3
    figure('name',names(i))
    for K = 1:k
        hold on
        plot3(data_set{i}(labels(i,:)' == K,1),data_set{i}(labels(i,:)' == K,2),data_set{i}(labels(i,:)' == K,3), '*','color',colors(K,:));
        hold off
    end

end

