angle = 15;
n = 360/angle;

[I, I_s] = rotate('data-source/sequence/apple/apple.png', angle);

imSaveAsPng('data-set/sequence/apple/apple', I_s, 1);
imSaveAsPng('data-set/sequence/apple_test/apple', I_s, 2:n);
imSaveAsGif('data-set/sequence/apple.gif', I_s);

figure('doublebuffer','on');
movie(immovie(I_s),10,40);

disp('done.');