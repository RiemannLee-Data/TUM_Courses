close all;
load('gesture_dataset.mat')

k=7
save_plots = false

Exercise3_kmeans( gesture_l, gesture_o, gesture_x, init_cluster_l,init_cluster_o,init_cluster_x, k)

if save_plots
    % save plots as pdfs
    f = gcf();
    while ~isempty(get(f,'name'))
        title = get(f,'name');
        filename = ['../SolutionSheet-Ex3/kmeans_' title '.pdf']
        print(f,filename,'-dpdf'); % then print it
        system(['pdfcrop ', filename])
        close(f);
        f = gcf();
    end
end

Exercise3_nubs( gesture_l, gesture_o, gesture_x, k )

if save_plots
    % save plots as pdfs
    f = gcf();
    while ~isempty(get(f,'name'))
        title = get(f,'name');
        filename = ['../SolutionSheet-Ex3/nubs_' title '.pdf']
        print(f,filename,'-dpdf'); % then print it
        system(['pdfcrop ', filename])
        close(f);
        f = gcf();
    end
    close(f)
end