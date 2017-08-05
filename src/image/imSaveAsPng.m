function [] = imSaveAsPng(filename, I_s, ind)

for n = ind
    I = I_s(:,:,:,n);
    imwrite(I,[filename sprintf('%04d', n) '.png'],'png');
end
  
end

