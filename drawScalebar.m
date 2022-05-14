function imoutput = drawScalebar(iminput,reallength,pxsize,side)
%DRAWSCALEBAR Summary of this function goes here
%   Detailed explanation goes here
[row,col,depth] = size(iminput);
barlength = round(reallength/pxsize);
percents = 0.05;
disp(pxsize)
switch side
    case 'upleft'
        rowcent = round(percents*row)+1;
        colcent = round(percents*col)+1;
    case 'upright'
        rowcent = round(percents*row)+1;
        colcent = round((1-percents)*col)+1-barlength;
    case 'downleft'
        rowcent = round((1-percents)*row)+1;
        colcent = round(percents*col)+1;
    case 'downright'
        rowcent = round((1-percents)*row)+1;
        colcent = round((1-percents)*col)+1-barlength;
end
if depth == 1
    imoutput = double(iminput)/max(double(iminput(:)));
    imoutput(rowcent:rowcent+10,colcent:colcent+barlength) = 1;
elseif depth ==3
    imoutput = double(iminput)/max(double(iminput(:)))*255;
    imoutput(rowcent:rowcent+10,colcent:colcent+barlength,:) = 255;
    imoutput = uint8(imoutput);
end

switch side
    case 'upleft'
        imoutput = insertText(imoutput,[colcent,rowcent+10+5],[num2str(round(reallength*1e6)),' microns'],...
            'Font','Calibri Bold','FontSize',32,'TextColor','white','BoxOpacity',0);
    case 'upright'
        rowcent = round(percents*row)+1;
        colcent = round((1-percents)*col)+1-barlength;
        imoutput = insertText(imoutput,[colcent,rowcent+10+5],[num2str(round(reallength*1e6)),' microns'],...
            'Font','Calibri Bold','FontSize',32,'TextColor','white','BoxOpacity',0);
    case 'downleft'
        rowcent = round((1-percents)*row)+1;
        colcent = round(percents*col)+1;
        imoutput = insertText(imoutput,[colcent,rowcent-40-5],[num2str(round(reallength*1e6)),' microns'],...
            'Font','Calibri Bold','FontSize',32,'TextColor','white','BoxOpacity',0);
    case 'downright'
        rowcent = round((1-percents)*row)+1;
        colcent = round((1-percents)*col)+1-barlength;
        imoutput = insertText(imoutput,[colcent,rowcent-40-5],[num2str(round(reallength*1e6)),' microns'],...
            'Font','Calibri Bold','FontSize',32,'TextColor','white','BoxOpacity',0);
end

end

