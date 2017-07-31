function [I, I_s] = rotate(filename, angle)

% read
[I,~,alpha] = imread(filename);
alpha = imresize(alpha, .2);
I = imresize(I, .2);

% mask
mask = (alpha ~= 0);
se = strel('disk',4);
mask = imclose(mask,se);

% center
[mcx, mcy] = find(bwmorph(mask,'shrink',inf));
xshift = size(I,1)/2-mcx(1);
yshift = size(I,2)/2-mcy(1);
mask = circshift(mask,int8([xshift yshift]));
I = circshift(I,int8([xshift yshift]));

% rotate
I_s = uint8(zeros([[size(I,1), size(I,2)], 3, floor(360/angle)]));
for n = 1:(360/angle)
     theta = (n-1)*angle;
     % rotate
     mask_rot = imrotate(mask,theta,'nearest','crop');
     I_rot = imrotate(I,theta,'bilinear','crop');
     % rgb
     for c = 1:3
         % cut
         I_cut = immultiply(mask_rot,I_rot(:,:,c));
         % insert
         I_empty = uint8(zeros([size(I,1), size(I,2)]));
         I_erased = immultiply(~mask_rot,I_empty);
         I_s(:,:,c,n) = imadd(I_erased,I_cut);
     end
end

end