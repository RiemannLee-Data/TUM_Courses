%% Multiple View Geometry 2020, Exercise Sheet 1
% Prof. Dr. Florian Bernard, Lukas Köstler, Mohammed Brahimi
% Some selfmade comments are added later
%% Exercise 1

% (b)
% load the image into the workspace
% note that we must you absolute path here
% otherwise the image will lose its color information
I = imread('lena.png');

% (c)
% determine the size of the image and show the image
[rows, cols, channels] = size(I) % denote the size of the image 
imshow(I), axis image

% (d)
% convert the image to gray scale and determine the maximum and minimum value of the image
I_gray = rgb2gray(I);

min_I = min(I_gray(:))
max_I = max(I_gray(:))

% (e)
% apply a Gaussian smoothing filter and save the output image
% construct a Gaussian kernel using the Matlab-function "fspecial"
% but not recommended, better use imgaussfilt or imgaussfilt3 instead
filt = fspecial('gaussian')
I_double = im2double(I_gray); % needed for convolution

I_filt = conv2(I_double,filt,'same');

imwrite(I_filt,'lena_gauss.png');

% (f)
%show the original image, the gray scale image and the filtered image in
%one figure and give the figures appropriater titles
subplot(131), imshow(I),      title('Original Lena')
subplot(132), imshow(I_gray), title('Grayscale Lena')
subplot(133), imshow(I_filt), title('Smoothed Lena')

% (g)
% compare the gray scale image and the filtered image for different values
% of the smoothing
% just choose other parameters in the fspecial function

%% Exercise 2

% (a) solve Ax=b for x
A = [2 2 0; 0 8 3]
b = [5; 15]

x = A\b

% (b) define a matrix B equal to A
B = A

% (c) change the second element in the first row of A
A(1,2) = 4

% (d) compute as instructed
c = 0;
for i=-4:4:4
    c = c + i*A'*b;
end
c % print c

% (e) compare the two operations and explain the difference
A.*B % is an element-wise multiplication, requires matrices of equal size
A'*B % is the matrix multiplication, requires number of columns of first factor = number of rows of second one

%% Exercise 3
% write a funcrtion to check if two vectors are almost equal
approxequal([1,2;3,4], [1,2;3.1,3.5], 0.5)

%% Exercise 4
% Write a function that return the sum of all prime numbers between and
% including s and e
addprimes(1,10)
addprimes2(1,10)