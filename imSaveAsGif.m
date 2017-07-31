function [] = imSaveAsGif(filename, I_s)

for n = 1:size(I_s,4)
    I = I_s(:,:,:,n);
    [imind,cm] = rgb2ind(I,256); 
    if n == 1 
      imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
    else 
      imwrite(imind,cm,filename,'gif','WriteMode','append'); 
    end 
end
  
end

