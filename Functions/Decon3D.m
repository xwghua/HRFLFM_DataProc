function [PSFpath,FLFMpath] = Decon3D(PSFpath,FLFMpath,Reconpath,Iter,Centers,dCenterPos,CutShift,CropH,CropW,Mode)
%DECON3D Summary of this function goes here
%   Detailed explanation goes here
load(PSFpath,'FLFPSF');
FLFPSF = single(FLFPSF);
Depth_Size = size(FLFPSF,3);
disp(['PSF [',PSFpath,'] read!',' Total layers: ',num2str(Depth_Size)])
%% Generate OTF
xsize = [size(FLFPSF,1), size(FLFPSF,2)];
msize = [size(FLFPSF,1), size(FLFPSF,2)];
mmid = floor(msize/2);
exsize = xsize + mmid;
exsize = [ min( 2^ceil(log2(exsize(1))), 128*ceil(exsize(1)/128) ), ...
           min( 2^ceil(log2(exsize(2))), 128*ceil(exsize(2)/128) ) ];
zeroImageEx = gpuArray(zeros(exsize, 'single'));
disp(['FFT size is ' num2str(exsize(1)) 'X' num2str(exsize(2))]);

OTF = gpuArray.zeros(size(zeroImageEx,1), size(zeroImageEx,2), size(FLFPSF,3),'single');
INVOTF = gpuArray.zeros(size(zeroImageEx,1), size(zeroImageEx,2), size(FLFPSF,3),'single');
OTF( 1:size(FLFPSF,1), 1:size(FLFPSF,2),:) = FLFPSF;
INVOTF( 1:size(FLFPSF,1), 1:size(FLFPSF,2),:) = imrotate(FLFPSF,180);
mc = 1 + [floor(size(FLFPSF,1)/2),floor(size(FLFPSF,2)/2)];
me = mc + exsize - 1;
exindices = {mod((mc(1):me(1))-1,  size(OTF,1))+1, mod((mc(2):me(2))-1,  size(OTF,2))+1};
OTF = fft2(OTF(exindices{:},:));
INVOTF = fft2(INVOTF(exindices{:},:));
OTF = OTF./max(OTF,[],[1,2]);
INVOTF = INVOTF./max(INVOTF,[],[1,2]);
extravar.INVOTF = INVOTF;
extravar.Centers = gpuArray(Centers);
extravar.dCenterPos = gpuArray(dCenterPos);
disp('************* OTF got! *************');
% disp({size(Centers),size(dCenterPos)})
%% ======================= Make Reconstruction Folder ==================================
FLFMfiles = dir([FLFMpath,'\*.tif']);
h = split(FLFMpath,'\');
% FLFMfolder = [char(h(end-1)),'_',char(h(end))];
FLFMfolder = char(h(end));
h = split(PSFpath,'\');
PSFfolder = char(h(end));
disp(["Reconstructions will be saved in";string([Reconpath,'\',PSFfolder(1:end-4),'-',FLFMfolder])])
mkdir([Reconpath,'\',PSFfolder(1:end-4),'-',FLFMfolder])
%% ======================= 3D Reconstruction Start ==================================
for ii = 1:1:length(FLFMfiles)
    imcut = gpuArray(single(imread([FLFMpath,'\',FLFMfiles(ii).name])));
    imcut = imcut/max(imcut(:))*60000;
    disp(['FLFM IMG [',FLFMfiles(ii).name,'] read!'])
    Xguess  = DeconvRL_3D_GPU_HUA( OTF, Iter, imcut, Mode ,extravar);
    Xguess = gather(Xguess);
    Xguess_norm = Xguess/max(Xguess(:))*65535;
%     Xguess  = DeconvRL_3D_GPU_HUA( OTF, Iter imcut, "fast" ,extravar);
%     Xguess = gather(Xguess);
%     Xguess_norm2 = Xguess/max(Xguess(:))*65535;
%     figure(1),imshowpair(Xguess_norm(:,:,80),Xguess_norm2(:,:,51),'montage')
    %% Save reconstructed images
    Xguess_norm_re = imresize( Xguess_norm , 275/123);
    Xguess_norm_re_proj = sum(Xguess_norm_re,3);
    rowc = round(size(Xguess_norm_re_proj,1)/2)+CutShift-250;
    colc = round(size(Xguess_norm_re_proj,2)/2)+CutShift;
    Xguess_norm_re = Xguess_norm_re(rowc-ceil(CropH/2)+1:rowc+floor(CropH/2),...
                                    colc-ceil(CropW/2)+1:colc+floor(CropW/2),:);
    t = Tiff([Reconpath,'\',PSFfolder(1:end-4),'-',FLFMfolder,'\',...
        FLFMfiles(ii).name(1:end-4),'iter',num2str(Iter),'.tif'],'w');
    tagstruct.ImageLength = size(Xguess_norm_re,1);
    tagstruct.ImageWidth = size(Xguess_norm_re,2);
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 16;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.Compression = Tiff.Compression.None;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    for depth_index  = 1:Depth_Size
        setTag(t,tagstruct);
        write(t,uint16(Xguess_norm_re(:,:,depth_index)));
        writeDirectory(t);
    end
    close(t);
end
end
