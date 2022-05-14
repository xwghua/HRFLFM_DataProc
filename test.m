% clear,clc
PSFpath = 'Z:\Xuanwen\FLFMuf\ExpData\Simu20200724Wv680gly\PSFFLFint_Sim65nm_20220320_Red_refine_gly_10um_1024.mat';
FLFMpath = 'D:\XW20220320\rawtif_selected_r_selected\rawtif_selected_r_4000_selected\';
% FLFMpath = 'Z:\Xuanwen\HRLF\GTdata\XW20200201\cell8flf_cut';

Reconpath = '.\';
Iter = 20;
Centers = [295,262;
           295,762;
           728,512];
dCenterPos = dcentpos;
CutShift = 75;
CropH = 1100;
CropW = 1100;

% Decon3D(PSFpath,FLFMpath,Reconpath,...
%         Iter,Centers,dCenterPos,...
%         CutShift,CropH,CropW);


%%
