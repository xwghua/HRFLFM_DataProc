function coordinates = roiSelect(imraw,imstackproj)
%ROISELECT Summary of this function goes here
%   Detailed explanation goes here
% imraw = zeros(300,300);
% imstackproj = zeros(300,300)+0.3;
global coord
imraw = single(imraw)/max(single(imraw(:)));
imstackproj = single(imstackproj)/max(single(imstackproj(:)));
fig = uifigure("Name","Select an ROI",...
               "WindowStyle","modal","Resize","off",...
               "Position",[500,500,620,330]);
ax1 = uiaxes('Parent',fig,'Position',[0,20,300,300]);
ax2 = uiaxes('Parent',fig,'Position',[300,20,300,300]);
axis(ax1,"tight");axis(ax2,"tight");
imshow(imraw,"Parent",ax1);imshow(imstackproj,"Parent",ax2);
uilabel("Text",'Row Center: ','Parent',fig,'Position',[40,10,100,22]);
uirowc = uieditfield(fig,'numeric','Position',[120,10,70,22],...
                     'Value',round(size(imstackproj,1)/2));
uilabel("Text",'Column Center: ','Parent',fig,'Position',[220,10,100,22]);
uicolc = uieditfield(fig,'numeric','Position',[320,10,70,22],...
                     'Value',round(size(imstackproj,2)/2));
uilabel("Text",'ROI Size: ','Parent',fig,'Position',[420,10,100,22]);
uisize = uieditfield(fig,'numeric','Position',[490,10,70,22],...
                     'Value',round(size(imstackproj,2)/2));
drawRect(ax2,ax2,imstackproj,uirowc.Value,uicolc.Value,uisize.Value);
uirowc.ValueChangedFcn = @(uiedtfield,event) ...
    drawRect(uirowc,ax2,imstackproj,uirowc.Value,uicolc.Value,uisize.Value);
uicolc.ValueChangedFcn = @(uiedtfield,event) ...
    drawRect(uicolc,ax2,imstackproj,uirowc.Value,uicolc.Value,uisize.Value);
uisize.ValueChangedFcn = @(uiedtfield,event) ...
    drawRect(uisize,ax2,imstackproj,uirowc.Value,uicolc.Value,uisize.Value);
fig.CloseRequestFcn = @(fig,event) fig2close(fig,[],[uirowc.Value,uicolc.Value,uisize.Value]);
uiwait(fig); 
coordinates = coord;
% coordinates = fig.CloseRequestFcn;
% coordinates = set(fig,'CloseRequestFcn', @(fig,event) fig2close(fig,[],[uirowc.Value,uicolc.Value]));
% fig.CloseRequestFcn = @(fig,event) fig2close(fig,[],[uirowc.Value,uicolc.Value]);
end

function drawRect(~,ax,im,rowc,colc,rectsize)
upper = max([round(rowc-rectsize/2),2]);
lower = min([round(rowc+rectsize/2),size(im,1)-1]);
left = max([round(colc-rectsize/2),2]);
right = min([round(colc+rectsize/2),size(im,2)-1]);
% disp([upper,lower,left,right])
linewidth = round((size(im,1)/100-1)/2);
im(upper-linewidth:upper+linewidth,left-linewidth:right+linewidth) = 1;
im(lower-linewidth:lower+linewidth,left-linewidth:right+linewidth) = 1;
im(upper-linewidth:lower+linewidth,left-linewidth:left+linewidth) = 1;
im(upper-linewidth:lower+linewidth,right-linewidth:right+linewidth) = 1;
imshow(im,"Parent",ax);
end

function fig2close(fig,~,coordinates)
global coord
coord = coordinates;
disp(coordinates);
delete(fig)
end