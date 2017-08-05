% code
addpath('../../src/image');

angle = 18;
n = 360/angle;

[I, I_s] = rotate('../../data/apple/apple.png', angle);

rmdir('dataset', 's');
mkdir('dataset/rotate/apple');
mkdir('dataset/rotate/apple_test');

imSaveAsPng('dataset/rotate/apple/apple', I_s, 1);
imSaveAsPng('dataset/rotate/apple_test/apple', I_s, 1:n);
imSaveAsGif('dataset/rotate/apple.gif', I_s);

figure('doublebuffer','on');
movie(immovie(I_s),10,40);

disp('done.');