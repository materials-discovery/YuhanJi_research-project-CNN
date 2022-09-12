function [z,zm] = GetZ2(ph)
% This script takes the Phase Field output from evpFFT and reconstructs the
% surface
ph(ph==2) = 0; % Make Binary
ph = permute(ph,[3 1 2]); % Rotate back
siz = size(ph);
z = zeros(siz(1),siz(2));
for i = 1:siz(1)
    for j = 1:siz(2)
        z(i,j) = find(ph(i,j,:)==1,1,'first');
    end
end


end

