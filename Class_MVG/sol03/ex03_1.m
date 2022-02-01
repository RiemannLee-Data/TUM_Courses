%% load model
[V,F] = openOFF('model.off', '');

%% close figure
close all;

%% rotate / translate vertices

figure
% figure creates a new figure window using default property values. The
% resulting figure is the current figure
% 简而言之就是每次transform都产生一个新的窗口，方便进行比较
display(V, F)
% refer to the end
% 注意这里的display是额外定义的函数，不是matlab内置的函数

% rotate first 45 degrees around x, then 120 degrees around z
figure
% also here the transform is defined outside
V1 = transform(V,  [deg2rad(45) 0 0           ]', [0 0 0]');
V1 = transform(V1, [0,          0 deg2rad(120)]', [0 0 0]');
display(V1, F)

% rotate first 120 degrees around z, then 45 degrees around x --> gives different 
% result, as the order in which rotations are applied matters
figure
V2 = transform(V,  [0           0 deg2rad(120)]', [0 0 0]');
V2 = transform(V2, [deg2rad(45) 0 0           ]', [0 0 0]');
display(V2, F)

% extend the function such that it can perform also translation
figure
V3 = transform(V,  [0 0 0]', [0.5 0.2 0.1]');
display(V3, F)

%% display model
function display(V,F)
    C = 0.3*ones(size(V,1),3);
    patch('Vertices', V, 'Faces', F, 'FaceVertexCData', C);
    axis equal;
    shading interp;
    camlight right;
    camlight left;
end