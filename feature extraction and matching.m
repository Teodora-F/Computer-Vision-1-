%QUESTION 2.1.1
%Reading the dom.jpg image
img_dom = rgb2gray(double(imread('dom.jpg'))/255.0);
%Adding different noises
dom_gauss = imnoise(img_dom,'gaussian', 0.3, 0.5);
dom_saltpep = imnoise(img_dom,'salt & pepper', 0.075);
dom_speckle = imnoise(img_dom,'speckle');

%Plotting the results
%{
subplot(2,2,1), imshow(img_dom,'InitialMagnification',800)
title('Original Image')
subplot(2,2,2), imshow(dom_gauss)
title('Gaussian noise')
subplot(2,2,3), imshow(dom_saltpep)
title('Salt and pepper noise')
subplot(2,2,4), imshow(dom_speckle)
title('Speckle noise')
%}


%QUESTION 2.1.2
%A pad of 4x4 is added at the borders
padded_matrix = zeros(608,458);
padded_matrix(5:604,5:454) = dom_saltpep;
[mat_max,mat_min,mat_mean,mat_median] = deal(padded_matrix);

size_dom = size(img_dom);
%Start iterating through every pixel and apply filter
for x=5:604 %row
   for y=5:454 %column
       list_neigh = []; %initialize list for the neighboring values around the center pixel
       mat_slice = padded_matrix(x-2:x+2,y-2:y+2); %obtain a sliced matrix matching the structuring element
       min_val = min(mat_slice(:));
       max_val = max(mat_slice(:));
       mean_val = mean(mat_slice(:))
       median_val = median(mat_slice(:));
   mat_max(x,y)= max_val; %replace 
   mat_min(x,y)= min_val;
   mat_median(x,y)= median_val;
   mat_mean(x,y)= mean_val;
   end
end


subplot(1,5,1), imshow(dom_saltpep)
title('Image with Salt&Pepper Noise')
subplot(1,5,2), imshow(mat_min)
title('Salt&Pepper Noise 5x5 min filter')
subplot(1,5,3), imshow(mat_max)
title('Salt&Pepper Noise 5x5 max filter')
subplot(1,5,4), imshow(mat_mean)
title('Salt&Pepper Noise 5x5 mean filter')
subplot(1,5,5), imshow(mat_median)
title('Salt&Pepper Noise 5x5 median filter')

%{
%QUESTION 2.2

%Using SURF detector to detect the features
regions1 = detectSURFFeatures(dom_saltpep);
regions2 = detectSURFFeatures(filter_used);
[features1, valid_corners1] = extractFeatures(dom_saltpep, regions1);
[features2, valid_corners2] = extractFeatures(filter_used, regions2);
%Get the matching features
indexPairs = matchFeatures(features1,features2); 
matchedPoints1 = valid_corners1(indexPairs(:,1),:); %get them separately
matchedPoints2 = valid_corners2(indexPairs(:,2),:);
nrpairs = size(indexPairs)
%Plot the matches
showMatchedFeatures(dom_saltpep,filter_used,matchedPoints1,matchedPoints2,'montage');
titlestr = sprintf('Feature matching between original image and min filter (%d matches)',nrpairs(1));
title(titlestr)

%Question 2.3
img1 = rgb2gray(double(imread('dom_full.jpeg'))/255.0);
img2 = rgb2gray(double(imread('dom_part.jpg'))/255.0);

det1  = detectSURFFeatures(img1);
det2 = detectSURFFeatures(img2);
[feat1, pts1] = extractFeatures(img1, det1);
[feat2, pts2] = extractFeatures(img2, det2);
indexPairs = matchFeatures(feat1, feat2);
match1  = pts1(indexPairs(:,1));
match2 = pts2(indexPairs(:,2));

figure;
showMatchedFeatures(img1,img2,match1,match2);
title('Matches (including outliers)');

%produce an estimate 2-D geometric transformation from the matched point
%where tform-matrix that maps the inliners between the matching points
%where inlierIdx- index of the inliners
[tform, inlierIdx] = estimateGeometricTransform2D(match2, match1, 'similarity');
inlierDistorted = match2(inlierIdx, :);
inlierOriginal  = match1(inlierIdx, :);

figure;
showMatchedFeatures(img1,img2,inlierOriginal,inlierDistorted);
title('Matches (inliers only)');

%Recover image by imwarp
outputView = imref2d(size(img1));
recov  = imwarp(img2,tform,'OutputView',outputView);
figure, imshowpair(img1,recov,'ColorChannels',[1 2 0])
title('Recovered image')

%}





