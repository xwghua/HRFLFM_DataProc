function imraw = shiftRaw(imraw,Centers,elmtSize,dCenterPos,resizing)
%SHIFTRAW Summary of this function goes here
%   Detailed explanation goes here
halfelmtSize = round(elmtSize/2);
imraw3d0 = cat(3,[],imraw(Centers(1,1)-halfelmtSize:Centers(1,1)+halfelmtSize,Centers(1,2)-halfelmtSize:Centers(1,2)+halfelmtSize),...
                 imraw(Centers(2,1)-halfelmtSize:Centers(2,1)+halfelmtSize,Centers(2,2)-halfelmtSize:Centers(2,2)+halfelmtSize),...
                 imraw(Centers(3,1)-halfelmtSize:Centers(3,1)+halfelmtSize,Centers(3,2)-halfelmtSize:Centers(3,2)+halfelmtSize));
imraw3d0bw = single(imraw3d0>0.3*max(imraw3d0(:)));

[X,Y] = meshgrid(1:1:(halfelmtSize*2+1),1:1:(halfelmtSize*2+1));
imcenters = [squeeze(mean(imraw3d0bw.*Y,[1,2])./mean(imraw3d0bw,[1,2])),squeeze(mean(imraw3d0bw.*X,[1,2])./mean(imraw3d0bw,[1,2]))];
imcenters = imcenters + [Centers(1,1)-halfelmtSize-1,Centers(1,2)-halfelmtSize-1;
                         Centers(2,1)-halfelmtSize-1,Centers(2,2)-halfelmtSize-1;
                         Centers(3,1)-halfelmtSize-1,Centers(3,2)-halfelmtSize-1];

D = (sqrt((imcenters(1,1)-imcenters(2,1))^2+(imcenters(1,2)-imcenters(2,2))^2) + ...
     sqrt((imcenters(1,1)-imcenters(3,1))^2+(imcenters(1,2)-imcenters(3,2))^2) + ...
     sqrt((imcenters(3,1)-imcenters(2,1))^2+(imcenters(3,2)-imcenters(2,2))^2) ) / 3 - halfelmtSize*2;

dCenterPos_re = dCenterPos + [Centers(1,1),Centers(1,2),Centers(2,1),Centers(2,2),Centers(3,1),Centers(3,2)];
D0 = (sqrt((dCenterPos_re(:,1)-dCenterPos_re(:,3)).^2+(dCenterPos_re(:,2)-dCenterPos_re(:,4)).^2) + ...
      sqrt((dCenterPos_re(:,1)-dCenterPos_re(:,5)).^2+(dCenterPos_re(:,2)-dCenterPos_re(:,6)).^2) + ...
      sqrt((dCenterPos_re(:,5)-dCenterPos_re(:,3)).^2+(dCenterPos_re(:,6)-dCenterPos_re(:,4)).^2) ) / 3 - halfelmtSize*2;

b = find(D0>D,1,"first");
a = find(D0<D,1,"last");

actual_dCenterPos = (D-D0(a))/(D0(b)-D0(a))*(dCenterPos(b,:)-dCenterPos(a,:)) + dCenterPos(a,:);
actual_dCenterPos = round(resizing*actual_dCenterPos);


imraw3d = cat(3,[],circshift(imresize(imraw3d0(:,:,1),resizing),-actual_dCenterPos(1:2)), ...
                   circshift(imresize(imraw3d0(:,:,2),resizing),-actual_dCenterPos(3:4)), ...
                   circshift(imresize(imraw3d0(:,:,3),resizing),-actual_dCenterPos(5:6)) );
imraw3d = imresize(imraw3d,1/resizing);
imraw(Centers(1,1)-halfelmtSize:Centers(1,1)+halfelmtSize,Centers(1,2)-halfelmtSize:Centers(1,2)+halfelmtSize) = imraw3d(:,:,1);
imraw(Centers(2,1)-halfelmtSize:Centers(2,1)+halfelmtSize,Centers(2,2)-halfelmtSize:Centers(2,2)+halfelmtSize) = imraw3d(:,:,2);
imraw(Centers(3,1)-halfelmtSize:Centers(3,1)+halfelmtSize,Centers(3,2)-halfelmtSize:Centers(3,2)+halfelmtSize) = imraw3d(:,:,3);
% figure(999),imshow(imraw/max(imraw(:)))
end

