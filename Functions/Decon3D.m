function [PSFpath,FLFMpath] = Decon3D(PSFpath,FLFMpath,Reconpath,Iter,Centers,...
    dCenterPos,...%CutShift,CropH,CropW,
    Mode,isreplace,varargin)
%DECON3D Summary of this function goes here
%   Detailed explanation goes here
load(PSFpath,'FLFPSF');
FLFPSF = single(FLFPSF);
FLFPSF = FLFPSF./max(FLFPSF,[],[1,2]);
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
first2recon = true;
for ii = 1:1:length(FLFMfiles)
    reconfilename = [Reconpath,'\',PSFfolder(1:end-4),'-',FLFMfolder,'\',...
        FLFMfiles(ii).name(1:end-4),'iter',num2str(Iter),'.tif'];
    procbar = round(ii/length(FLFMfiles)*100,2);
    if ~exist(reconfilename,"file") || isreplace
        imcut = gpuArray(single(imread([FLFMpath,'\',FLFMfiles(ii).name])));
        imcut = imcut/max(imcut(:))*60000;
        disp(['FLFM IMG [',FLFMfiles(ii).name,'] read!'])
        if nargin>8
            varargin{1}.Text = ['(',num2str(procbar),'%) ',char(Mode),...
                ': Reconstructing FLFM IMG [',FLFMfiles(ii).name,']'];
            varargin{1}.BackgroundColor = [0.85,0.41,0.41];
            drawnow;
        end
        Xguess  = DeconvRL_3D_GPU_HUA( OTF, Iter, imcut, Mode ,extravar);
        Xguess = gather(Xguess);
        Xguess_norm = Xguess/max(Xguess(:))*65535;
        Xguess_norm_re = imresize( Xguess_norm , 275/123);
        Xguess_norm_re_proj = sum(Xguess_norm_re,3);
        %% Save reconstructed images
        if first2recon
            coord = roiSelect(imcut,Xguess_norm_re_proj);
            first2recon = ~first2recon;
        end
        rowc = coord(1);
        colc = coord(2);
        Xsize = coord(3);        
        if nargin>8
            Xguess_norm_re = Xguess_norm_re(rowc-ceil(Xsize/2)+1:rowc+floor(Xsize/2),...
                colc-ceil(Xsize/2)+1:colc+floor(Xsize/2),:);
            Xguess_norm_re_proj = sum(Xguess_norm_re,3);
            imshowpair(imcut/max(imcut(:)),Xguess_norm_re_proj/max(Xguess_norm_re_proj(:)),...
                'montage','parent',varargin{2});
            drawnow
            % ==============================
            varargin{1}.Text = ['(',num2str(procbar),'%) ',char(Mode),...
                ': Saving FLFM IMG [',FLFMfiles(ii).name,']'];
            varargin{1}.BackgroundColor = [0.41,0.41,0.85];
            drawnow;
        end
        t = Tiff(reconfilename,'w');
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
    else
%         disp(['FLFM IMG [',FLFMfiles(ii).name,'] exists! No replacement!'])
        if nargin>8
            varargin{1}.Text = ['(',num2str(procbar),'%) ','FLFM IMG [',...
                FLFMfiles(ii).name,'] exists! No replacement!'];
            varargin{1}.BackgroundColor = [0.85,0.65,0];
            drawnow;
        end
    end
end
end
