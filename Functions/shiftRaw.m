function imraw = shiftRaw(imraw,Centers,elmtSize,dCenterPos)
%SHIFTRAW Summary of this function goes here
%   Detailed explanation goes here

imraw3d0 = cat(3,imraw(Centers(1,1)-elmtSize:Centers(1,1)+elmtSize,Centers(1,2)-elmtSize:Centers(1,2)+elmtSize),...
                 imraw(Centers(2,1)-elmtSize:Centers(2,1)+elmtSize,Centers(2,2)-elmtSize:Centers(2,2)+elmtSize),...
                 imraw(Centers(3,1)-elmtSize:Centers(3,1)+elmtSize,Centers(3,2)-elmtSize:Centers(3,2)+elmtSize));
imraw3d0bw = single(imraw3d0>0.3*max(imraw3d0(:)));

[X,Y] = meshgrid(1:1:(elmtSize*2+1),1:1:(elmtSize*2+1));
imcenters = [squeeze(mean(imraw3d0bw.*Y,[1,2])./mean(imraw3d0bw,[1,2])),squeeze(mean(imraw3d0bw.*X,[1,2])./mean(imraw3d0bw,[1,2]))];

D = (sqrt((imcenters(1,1)-imcenters(2,1))^2+(imcenters(1,2)-imcenters(2,2))^2) + ...
     sqrt((imcenters(1,1)-imcenters(3,1))^2+(imcenters(1,2)-imcenters(3,2))^2) + ...
     sqrt((imcenters(3,1)-imcenters(2,1))^2+(imcenters(3,2)-imcenters(2,2))^2) ) / 3 - elmtSize;

D0 = (sqrt((dCenterPos(:,1)-dCenterPos(:,3))^2+(dCenterPos(:,2)-dCenterPos(:,4))^2) + ...
      sqrt((dCenterPos(:,1)-dCenterPos(:,5))^2+(dCenterPos(:,2)-dCenterPos(:,6))^2) + ...
      sqrt((dCenterPos(:,5)-dCenterPos(:,3))^2+(dCenterPos(:,6)-dCenterPos(:,4))^2) ) / 3;

b = find(D0>D,1,"first");
a = find(D0<D,1,"last");

actual_dCenterPos = (D-D0(a))/(D0(b)-D0(a))*(dCenterPos(b)-dCenterPos(a)) + dCenterPos(a);

imraw3d = circshift(imraw3d0(:,:,1),fliplr(-actual_dCenterPos(1:2))) + ...
          circshift(imraw3d0(:,:,2),fliplr(-actual_dCenterPos(3:4))) + ...
          circshift(imraw3d0(:,:,3),fliplr(-actual_dCenterPos(5:6)));
imraw(Centers(1,1)-elmtSize:Centers(1,1)+elmtSize,Centers(1,2)-elmtSize:Centers(1,2)+elmtSize) = imraw3d(:,:,1);
imraw(Centers(2,1)-elmtSize:Centers(2,1)+elmtSize,Centers(2,2)-elmtSize:Centers(2,2)+elmtSize) = imraw3d(:,:,2);
imraw(Centers(3,1)-elmtSize:Centers(3,1)+elmtSize,Centers(3,2)-elmtSize:Centers(3,2)+elmtSize) = imraw3d(:,:,3);
end

