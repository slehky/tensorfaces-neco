function hText = RotateXLabel(angleRotation,newLabels)
%ROTATEXLABEL - Rotate XTickLabels by some angle
%
% Syntax: hText = RotateXLabel(angleRotation,newLabels);
%
% Inputs:
% angleRotation - A number between -180 and +180 degrees
% newLabels - Cell or character array containing the desired xticklabels
%
% Output:
% hText - handle for the xticklabel text objects
%
% Example 1: Rotate labels by +45 degrees
% plot(1:5); set(gca,'xtick',(1:5))
% newLabels = {'One','Two','Three','Four','Five'};
% hText = RotateXLabel(45,newLabels);
%
% Example 2: Rotate labels by -90 degrees
% plot(1:5); set(gca,'xtick',(1:5))
% newLabels = {'One','Two','Three','Four','Five'};
% hText = RotateXLabel(-90,newLabels);
%
% See also: XTICKLABEL_ROTATE90 (Mathworks File Exchange)
%
% Denis Gilbert, CSSM Post 08-May-2003


%Check the input arguments
if abs(angleRotation) > 180
   error('DEGREES should be between -180 and 180');
end


if nargin == 1
   newLabels = get(gca,'XTickLabel');
elseif nargin == 0 | nargin > 2
   error('RotateXLabel requires 1 or 2 input arguments');
end


xtl = get(gca,'XTickLabel');
set(gca,'XTickLabel','');
lxtl = length(xtl);
if nargin > 1
    lnl = length(newLabels);
    if lnl~=lxtl
        error('Number of new labels must equal number of old');
    end;
    xtl = newLabels;
end;


yLim = get(gca,'YLim');
hxLabel=get(gca,'XLabel');
xLP=get(hxLabel,'Position');


%Optimize space by bringing the xtick labels closer to the bottom x-axis
y = xLP(2) +0.4 * (yLim(1) - xLP(2));


XTick=get(gca,'XTick');
y=repmat(y,length(XTick),1);
fs=get(gca,'fontsize');
hText=text(XTick,y,xtl,'fontsize',fs);


%Set the horizontal alignment differently for positive and negative angles
if angleRotation >= 0
   set(hText,'Rotation',angleRotation,'HorizontalAlignment','right');
elseif angleRotation < 0
   set(hText,'Rotation',angleRotation,'HorizontalAlignment','left');
end


%------------- END OF CODE --------------