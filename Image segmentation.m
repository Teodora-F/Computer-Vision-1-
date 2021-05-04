%QUESTION 3.1
%Read Image
img = imread('cells.png');
img_grey = rgb2gray(double(imread('cells.png'))/255.0);
img_grey_sing = rgb2gray(img);

%OTSU's method
lev = graythresh(img_grey);
otsu_img = imbinarize(img_grey,lev);

%Adaptive thresholding method
T = adaptthresh(img_grey, 0.4); %sensitivity parameter
adapt_img = imbinarize(img_grey,T);

%K-means method
[L,Centers] = imsegkmeans(img_grey_sing,3); %number of clusters desired
k_img = labeloverlay(img_grey_sing,L);

%Edge detection method - morphology segmentation
%Get threshold value of edges using Sobel operator
[~,threshold] = edge(img_grey,'sobel');
factor = 0.8; %check for optimal 
edge_img = edge(img_grey,'sobel',threshold * factor);

%Dilate using structuring elements
e90 = strel('line',2,90);
e0 = strel('line',2,0);
dil_img = imdilate(edge_img,[e90 e0]);

%Filling the gaps
fill_img = imfill(dil_img,'holes');

%Cleaning the borders of structures that are connected to the edges
noborders_img = imclearborder(fill_img,8);

elemd = strel('diamond',1);
img_edge = imerode(noborders_img,elemd);
img_edge = imerode(img_edge,elemd);
%figure;imshow(labeloverlay(img_grey,img_final))

%Watershade method
img_filter = adapthisteq(img_grey); %contrast limited adaptive histogram equalization
level = graythresh(img_grey);
img_bin = imbinarize(img_grey,level);
img_dist = -bwdist(~img_bin); %Euclidean distance transform
img_dist(~img_bin) = -Inf;
img_wat = watershed(img_dist);
%wat_final = imshow(label2rgb(img_wat,'jet','w'))

%Plotting the data
%{
fig1 = figure;
imshow(img_filter)
hold on
fig_otsu = imshow(otsu_img)
fig_otsu.AlphaData = 0.25
title('OTSU method')

fig2 = figure;
imshow(img_filter)
hold on
fig_bin = imshow(adapt_img)
fig_bin.AlphaData = 0.3
title('Adaptive thresholding method')

fig3 = figure;
imshow(img_filter)
hold on
fig_k = imshow(k_img)
fig_k.AlphaData = 0.3
title('K-Means method')

fig4 = figure;
imshow(img_filter)
hold on
fig_edge = imshow(img_edge)
fig_edge.AlphaData = 0.3
title('Edge detection method')

fig5 = figure;
imshow(img_filter)
hold on
wat_final = imshow(label2rgb(img_wat,'jet','w'))
wat_final.AlphaData = 0.3
title('Watershed method')
%}


%QUESTION 3.2
img_outline = bwperim(img_wat);
segment = img_grey; 
segment(img_outline) = 255; 
%figure;imshow(segment)

dilatedImage = imdilate(segment,strel('disk',1));
se = strel('disk',1);
finalsmooth = imclose(dilatedImage,se);
%imshow(finalsmooth)
imshowpair(segment,finalsmooth, 'montage')

%Failed attempt at solving oversegmentation
%{
mask = imextendedmin(img_dist,2);
figure;imshowpair(img_grey,mask,'blend')
D2 = imimposemin(img_dist,mask,8);
Ld2 = watershed(D2);
bw3 = img_grey;
bw3(Ld2 == 0) = 255;
figure;imshow(bw3)
%}

%QUESTION 3.3
%3.3.1
otsu_bin = imbinarize(img_grey,lev);
area_cells = table2array(regionprops('table',otsu_bin,img_grey,'Area'))
%3.3.2
greench = img(:, :, 2); 
green_grey = im2gray(greench);
levgreen = graythresh(green_grey);
otsu_green = imbinarize(img_grey,levgreen);
bright_cells = table2array(regionprops('table',otsu_green,green_grey,'MeanIntensity'));
%3.3.3
mean_area = mean(area_cells)
std_area = std(area_cells)
mean_bright = mean(bright_cells)
std_bright = std(bright_cells)

%QUESTION 3.4

img_ecoli = imread('Ecoli.png');
img_grey_ecoli = rgb2gray(double(imread('Ecoli.png'))/255.0);

%Convert to L*a*b* colorspace 
img_lab = rgb2lab(img_ecoli);
ab = img_lab(:,:,2:3); %Choose only color related channels, ignore the brightness one
ab = im2single(ab);

colors_use = 4; %number of clusters based on colours we want to create, it can be seen in img_lab that there are more than 3 colours
pixel_labels = imsegkmeans(ab,colors_use,'NumAttempts',3);
mask1 = pixel_labels==4; %mask of Green colour, most common
clust_green = img_ecoli .* uint8(mask1);
figure;imshowpair(img_ecoli,clust_green,'montage')
title('Segmented GFP cells')

clust_gray = rgb2gray(clust_green);
img_outline_ecoli = bwperim(clust_gray);
segment_ecoli = img_grey_ecoli; 
segment_ecoli(img_outline_ecoli) = 255; 
%figure;imshow(segment_ecoli)

dilatedecoli = imdilate(segment_ecoli,strel('disk',1));
se = strel('disk',1);
finalecoli = imclose(dilatedecoli,se);
figure;imshowpair(segment_ecoli,finalecoli,'montage')
title('Outlined cells')






