function [M11, M12, M22] = getM(I, sigma)
% Compute structure tensor

% spatial gradient using central differences
Ix = 0.5*(I(:,[2:end end]) - I(:,[1 1:end-1]));
Iy = 0.5*(I([2:end end],:) - I([1 1:end-1],:));

% Gaussian kernel
% 该函数对输入元素取整，返回不小于输入值的最小整数
% 对于复数则分别对实部和虚部取整
k = ceil(4*sigma+1);
G = fspecial('gaussian', k, sigma);

M11 = conv2(Ix .* Ix, G, 'same');
M12 = conv2(Ix .* Iy, G, 'same');
M22 = conv2(Iy .* Iy, G, 'same');

% Note: We do not need to compute M21 sparately, since M is symmetrical and
% therefore M21 == M12

end
