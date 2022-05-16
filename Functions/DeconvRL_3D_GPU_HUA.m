function Xguess = DeconvRL_3D_GPU_HUA( OTF, maxIter, FLFimg, mode, varargin)

if mode == "hybrid"
    projfunc = @BackForewardProj_hybrid;
    halfelmtSize = round((varargin{1}.Centers(2,2)-varargin{1}.Centers(1,2))/2);
    [X,Y] = meshgrid(1:1:(halfelmtSize*2+1),1:1:(halfelmtSize*2+1));
    extravar.circmask = gpuArray(single(sqrt((X-halfelmtSize).^2 + (Y-halfelmtSize).^2)<=250));
    extravar.resizeParam = ceil(max(abs(varargin{1}.dCenterPos(:))));
    extravar.resizeFactor = 1;
    extravar.elmtSize = halfelmtSize;
    extravar.Centers = varargin{1}.Centers;
    extravar.dCenterPos = varargin{1}.dCenterPos;
    extravar.INVOTF = varargin{1}.INVOTF;
elseif mode == "shifted"
    projfunc = @BackForewardProj_hybrid;
    halfelmtSize = round((varargin{1}.Centers(2,2)-varargin{1}.Centers(1,2))/2);
    [X,Y] = meshgrid(1:1:(halfelmtSize*2+1),1:1:(halfelmtSize*2+1));
    extravar.circmask = gpuArray(single(sqrt((X-halfelmtSize).^2 + (Y-halfelmtSize).^2)<=halfelmtSize));
    extravar.resizeParam = ceil(max(abs(varargin{1}.dCenterPos(:))));
    extravar.resizeFactor = 1;
    extravar.elmtSize = halfelmtSize;
    extravar.Centers = varargin{1}.Centers;
    extravar.dCenterPos = varargin{1}.dCenterPos;
    extravar.INVOTF = varargin{1}.INVOTF;
    try
        FLFimg = shiftRaw(FLFimg,extravar.Centers,halfelmtSize*2,extravar.dCenterPos,10);
    catch
        warning('No rawshift appliable! Please check the raw data!')
    end
elseif mode == "wave"
    projfunc = @BackForewardProj_waveopt;
    extravar = varargin{1}.INVOTF;
elseif mode == "fast"
    projfunc = @BackForewardProj_waveopt_fast;
    extravar = varargin{1}.INVOTF;
end

time_sum = 0;
Xguess = gpuArray.ones(size(FLFimg,1),size(FLFimg,2),size(OTF,3));
Project_Error = FLFimg;
tic;
[Xguess,Project_Error] = projfunc(OTF,Xguess,FLFimg,Project_Error,extravar);
time_sum = time_sum + toc;
for ii = 2:maxIter
    tic;
    [Xguess,Project_Error] = BackForewardProj_waveopt_fast(OTF,Xguess,FLFimg,Project_Error,varargin{1}.INVOTF);
    time_sum = time_sum + toc;
end
disp(['iter ' num2str(ii) '|' num2str(maxIter) ' *DONE*, single: ' num2str(time_sum/maxIter) ' secs, ' ...
        ' total: ' num2str(time_sum) ' secs.']);
end

function [Xguess,ForeProjErr] = BackForewardProj_hybrid(OTF,Xguess,FLFimg,ProjErr,extravar)
elmtSize = extravar.elmtSize;
Centers = extravar.Centers;
circmask = extravar.circmask;
resizeParam = extravar.resizeParam;
dCenterPos = round(extravar.dCenterPos);
ForeProj = gpuArray.zeros(size(FLFimg,1), size(FLFimg,2),'single');
FLFimg3d0 = cat(3,circmask.*ProjErr(Centers(1,1)-elmtSize:Centers(1,1)+elmtSize,...
                                    Centers(1,2)-elmtSize:Centers(1,2)+elmtSize),...
                  circmask.*ProjErr(Centers(2,1)-elmtSize:Centers(2,1)+elmtSize,...
                                    Centers(2,2)-elmtSize:Centers(2,2)+elmtSize),...
                  circmask.*ProjErr(Centers(3,1)-elmtSize:Centers(3,1)+elmtSize,...
                                    Centers(3,2)-elmtSize:Centers(3,2)+elmtSize));    
FLFimg3d0 = cat(2,zeros(size(FLFimg3d0,1),resizeParam,size(FLFimg3d0,3)),FLFimg3d0,...
                  zeros(size(FLFimg3d0,1),resizeParam,size(FLFimg3d0,3)));
FLFimg3d0 = cat(1,zeros(resizeParam,size(FLFimg3d0,2),size(FLFimg3d0,3)),FLFimg3d0,...
                  zeros(resizeParam,size(FLFimg3d0,2),size(FLFimg3d0,3)));
for cc = 1:size(OTF,3)
    FLFimg3d = circshift(FLFimg3d0(:,:,1),fliplr(-dCenterPos(cc,1:2))) + ...
               circshift(FLFimg3d0(:,:,2),fliplr(-dCenterPos(cc,3:4))) + ...
               circshift(FLFimg3d0(:,:,3),fliplr(-dCenterPos(cc,5:6)));
    FLFimg3d = cat(2,zeros(size(FLFimg3d,1),round((size(FLFimg,2)-size(FLFimg3d,2))/2)),FLFimg3d,...
                     zeros(size(FLFimg3d,1),size(FLFimg,2)-size(FLFimg3d,2)-round((size(FLFimg,2)-size(FLFimg3d,2))/2)));
    FLFimg3d = cat(1,zeros(round((size(FLFimg,1)-size(FLFimg3d,1))/2),size(FLFimg3d,2)),FLFimg3d,...
                     zeros(size(FLFimg,1)-size(FLFimg3d,1)-round((size(FLFimg,1)-size(FLFimg3d,1))/2),size(FLFimg3d,2)));
    Xguess(:,:,cc) = Xguess(:,:,cc) .*FLFimg3d;
    Xguess(isnan(Xguess(:,:,cc)))=0;
    Xguess(:,:,cc) = Xguess(:,:,cc) + max(max(Xguess(:,:,cc)))*0.001;
    FLFimg3d = real(OTF(:,:,1)) * 0;
    FLFimg3d(1:size(FLFimg,1),1:size(FLFimg,2)) = Xguess(:,:,cc);
    tempProj = real(ifft2(OTF(:,:,cc).*fft2((FLFimg3d))));
    ForeProj = ForeProj + tempProj(1:size(FLFimg,1),1:size(FLFimg,2));
end
ForeProjErr = FLFimg./ForeProj;
ForeProjErr(isnan(ForeProjErr))=0;
ForeProjErr = ForeProjErr + max(ForeProjErr(:))*0.001;
end

function [Xguess,ForeProjErr] = BackForewardProj_waveopt(OTF,Xguess,FLFimg,ProjErr,extravar)
ProjErrPad = real(OTF(:,:,1))*0;
ProjErrPad(1:size(FLFimg,1),1:size(FLFimg,2)) = ProjErr;
ForeProj = gpuArray.zeros(size(FLFimg,1), size(FLFimg,2),'single');
for cc = 1:size(OTF,3)
    tempProj = real(ifft2(extravar(:,:,cc).*fft2(ProjErrPad)));
    Xguess(:,:,cc) = Xguess(:,:,cc) .*tempProj(1:size(FLFimg,1),1:size(FLFimg,2));
    Xguess(isnan(Xguess(:,:,cc)))=0;
    Xguess(:,:,cc) = Xguess(:,:,cc) + max(max(Xguess(:,:,cc)))*0.001;
    tempProj = tempProj*0;
    tempProj(1:size(FLFimg,1),1:size(FLFimg,2)) = Xguess(:,:,cc);
    tempProj = real(ifft2(OTF(:,:,cc).*fft2(tempProj)));
    ForeProj = ForeProj + tempProj(1:size(FLFimg,1),1:size(FLFimg,2));
end
ForeProjErr = FLFimg./ForeProj;
ForeProjErr(isnan(ForeProjErr))=0;
ForeProjErr = ForeProjErr + max(ForeProjErr(:))*0.001;
end

function [Xguess,ForeProjErr] = BackForewardProj_waveopt_fast(OTF,Xguess,FLFimg,ProjErr,extravar)
ProjErrPad = real(OTF(:,:,1))*0;
ProjErrPad(1:size(FLFimg,1),1:size(FLFimg,2)) = ProjErr;
tempProj = real(ifft2(extravar.*fft2(ProjErrPad)));
Xguess = Xguess .*tempProj(1:size(FLFimg,1),1:size(FLFimg,2),:);
Xguess(isnan(Xguess))=0;
Xguess = Xguess + max(Xguess(:))*0.001;
tempProj = tempProj*0;
tempProj(1:size(FLFimg,1),1:size(FLFimg,2),:) = Xguess;
tempProj = real(ifft2(OTF.*fft2(tempProj)));
ForeProjErr =FLFimg./sum(tempProj(1:size(FLFimg,1),1:size(FLFimg,2),:),3);
ForeProjErr(isnan(ForeProjErr))=0;
ForeProjErr = ForeProjErr + max(ForeProjErr(:))*0.001;
end