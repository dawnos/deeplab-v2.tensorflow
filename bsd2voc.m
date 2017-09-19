
addpath('/mnt/DataBlock2/VOCdevkit/VOCcode');

bsdRoot = '/mnt/DataBlock2/inst';
vocRoot = '/mnt/DataBlock2/VOCdevkit/VOC2012/SegmentationClassAug';

files = dir([bsdRoot '/*.mat']);

colormap = VOClabelcolormap(256);

for i = 1:length(files)
  load([bsdRoot '/' files(i).name]);
  seg = zeros(size(GTinst.Segmentation));
  for c = 1:length(GTinst.Categories)
    seg(GTinst.Segmentation == c) = GTinst.Categories(c);
  end
  
  imwrite(seg, colormap, [vocRoot '/' files(i).name(1:(end-3)) 'png']);
  % imagesc(seg);
  % waitforbuttonpress;
end