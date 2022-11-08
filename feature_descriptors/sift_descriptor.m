format long

f=imread('out90.png');
% Change sigma for different standard deviations e.g. 0.25, 0.5, 0.75 etc.
sigma = 0.5;

w2 = fspecial('log',[3 3], sigma); 

% % Returns a rotationally symmetric Laplacian of Gaussian filter of size hsize with standard deviation sigma

filtered_img2=imfilter(f,w2,'replicate'); 
figure
imshow(filtered_img2);
title("Filtered Image")

% Detect scale invariant feature transform (SIFT) features and return SIFTPoints object
f2= rgb2gray(f);
filtered_img2 = rgb2gray(filtered_img2);
points = detectSIFTFeatures(f2);
points_2 = detectSIFTFeatures(filtered_img2);

% Overlay most salient features on Image
figure
imshow(f);
hold on;
plot(points.selectStrongest(200));
title("Most Salient Features Overlayed on Image")

% Overlay 200 most salient features on filtered image
figure
imshow(filtered_img2);
hold on;
plot(points_2.selectStrongest(200));
title("Most Salient Features Overlayed on Filtered Image")

%% Feature Matching Diagram Example
% Convert images to grayscale
out93 = imread("out93.png");
out94 = imread("out94.png");
out93gray = rgb2gray(out93);
out94gray = rgb2gray(out94);

% RANSAC feature matching 
% ContrastThreshold for selecting the strongest features, specified as a 
% non-negative scalar in the range [0,1]. The threshold is used to filter 
% out weak features in low-contrast regions of the image. Increase the 
% contrast threshold to decrease the number of returned features, default
% 0.0133

% Tuning mask parameters and stopping criteriion will improve matching
% results. 
% load stereoPointPairs
% %SIFT
% points93SIFT = detectSIFTFeatures(out93gray, ContrastThreshold=0.02, Sigma = 1.5, EdgeThreshold = 10); % Detect SIFT features
% points93SIFT = points93SIFT.selectStrongest(500); % Select 400 most salient
% points94SIFT = detectSIFTFeatures(out94gray,  ContrastThreshold=0.02, Sigma = 1.5, EdgeThreshold = 10); % Detect SIFT Features
% points94SIFT = points94SIFT.selectStrongest(500); % Select 400 most salient
% 
% 
% figure;
% showMatchedFeatures(out93,out94,points93SIFT,points94SIFT);
% title("SIFT Matched Features")
% % Compute fundamental matrix
% % Due to element of randomness in RANSAC method, script may have to be run
% % multiple times to get output if there arnt enough overlapping features
% % chosen
% fRANSAC = estimateFundamentalMatrix(points93SIFT, ...
%     points94SIFT,Method="RANSAC", ...
%     NumTrials=2000,DistanceThreshold=1e-4);
m = size(out94gray);
% gradients = (points93SIFT.Location(:,2)-points94SIFT.Location(:,2))./(points93SIFT.Location(:,1)-points94SIFT.Location(:,1));
% gradients_std = std(gradients);
% gradients_mean = mean(gradients);
% figure;
% showMatchedFeatures(out93,out94,points93SIFT, points94SIFT,'montage','PlotOptions',{'ro','go','y--'});
% title('Putative Point Matches, SIFT');
% hold on
% 
% for i=1:length(gradients)
%     % plot feature matches that gradient between points exceeds gradients
%     % mean + 0.5 * std gradients, as correctly matched features should be
%     % ~parrallel.
%     if (abs(gradients(i)) >= (gradients_mean + 0.5*gradients_std))
%         plot([points93SIFT.Location(i,1), points94SIFT.Location(i,1) + m(2)],[points93SIFT.Location(i,2), points94SIFT.Location(i,2)],'-b')
%     end
% 
% end
% 
% %% Harris
% points93Harris = detectHarrisFeatures(out93gray); % Detect Harris corners
% points94Harris = detectHarrisFeatures(out94gray); 
% [HarrisFeatures1,Harris_valid_points1] = extractFeatures(out93gray,points93Harris); % Extract Neighbourhood features
% [HarrisFeatures2,Harris_valid_points2] = extractFeatures(out94gray,points94Harris); 
% indexPairs = matchFeatures(HarrisFeatures1,HarrisFeatures2); % Match the features
% HarrisMatchedPoints1 = Harris_valid_points1(indexPairs(:,1),:);
% HarrisMatchedPoints2 = Harris_valid_points2(indexPairs(:,2),:);
% 
% m = size(out94gray);
% harrisGradients = (HarrisMatchedPoints1.Location(:,2)-HarrisMatchedPoints2.Location(:,2))./(HarrisMatchedPoints1.Location(:,1)-HarrisMatchedPoints2.Location(:,1));
% harrisGradientsStd = std(harrisGradients);
% harrisGradientsMean = mean(harrisGradients);
% 
% figure; 
% showMatchedFeatures(out93,out94,HarrisMatchedPoints1,HarrisMatchedPoints2);
% title("Harris Corner Detection");
% 
% figure;
% showMatchedFeatures(out93,out94,HarrisMatchedPoints1, HarrisMatchedPoints2,'montage','PlotOptions',{'ro','go','y--'});
% title("Harris Corner Detection")
% hold on;
% for i=1:length(gradients)
%     % plot feature matches that gradient between points exceeds gradients
%     % mean + 0.5 * std gradients, as correctly matched features should be
%     % ~parrallel.
%     if (abs(harrisGradients(i)) >= (harrisGradientsMean + 2*harrisGradientsStd))
%         plot([HarrisMatchedPoints1.Location(i,1), HarrisMatchedPoints2.Location(i,1) + m(2)],[HarrisMatchedPoints1.Location(i,2), HarrisMatchedPoints2.Location(i,2)],'-b')
%     end
% 
% end
% legend("Matched", "Incorrectly Matched")

%% SURF, showing invariance to rotation and translation
I1 = rgb2gray(imread('out90.png'));
I2 = imresize(imrotate(I1,-20),1.2);
SURFpoints1 = detectSURFFeatures(I1);
SURFpoints2 = detectSURFFeatures(I2);
[SURFFeatures1, SURFvpts1] = extractFeatures(I1,SURFpoints1);
[SURFFeatures2, SURFvpts2] = extractFeatures(I2,SURFpoints2);
indexPairs = matchFeatures(SURFFeatures1,SURFFeatures2) ;
SURFMatchedPoints1 = SURFvpts1(indexPairs(:,1));
SURFMatchedPoints2 = SURFvpts2(indexPairs(:,2));
figure; showMatchedFeatures(I1,I2,SURFMatchedPoints1,SURFMatchedPoints2);
legend('matched points 1','matched points 2');
title("SURF Feature Matching")
%%
% figure
% subfigure(4,1,1)
% showMatchedFeatures(out93,out94,HarrisMatchedPoints1, HarrisMatchedPoints2,'montage','PlotOptions',{'ro','go','y--'});
% title("Harris Corner Detection")
% subfigure(4,1,2)
% showMatchedFeatures(I1,I2,SURFMatchedPoints1,SURFMatchedPoints2);
% legend('matched points 1','matched points 2');
% title("SURF Feature Matching")
% subfigure(4,1,3)

%% FAST
FASTPoints1 = detectFASTFeatures(out93gray,'minContrast',15/255,'minQuality',1/255);
FASTPoints2 = detectFASTFeatures(out94gray,'minContrast',15/255,'minQuality',1/255);

h = fspecial('gauss',5);
Ig1 = imfilter(out93gray,h);
Ig2 = imfilter(out94gray,h);
corners1 = detectFASTFeatures(Ig1,'minContrast',15/255,'minQuality',1/255)
corners2 = detectFASTFeatures(Ig2,'minContrast',15/255,'minQuality',1/255)
locs1 = corners1.Location;
locs2 = corners1.Location;
figure;
for ii = 1:size(locs1,1)
    out93(floor(locs1(ii,2)),floor(locs1(ii,1)),2) = 255; % green dot
end
hold on;
for ii = 1:size(locs1,1)
plot(locs1(ii,2),locs1(ii,1), 'xr')
end
imshow(out93)
title("FAST Corner Detection ")
%% 
function F = getFundamentalMatrix(A)
[U, S, V] = svd(A);
x = V(:, end);
F = reshape(x, [3,3])';
end

function [E_true, iE, nmax] = cheirality(E, p, q, K1, K2)
% Function: Return the true essential matrix in a bunch of estimates
% 
% The cheirality check is an operation to eliminate false solutions of
% essential matrices computed using 5-point algorithm or 7-point algorithm
% by reprojecting 3D points back to the associated camera positions. The
% solution with most points in front of both cameras is considered as the
% true solution
% 
% Usage:
%       [E_true, iE, nmax] = cheirality(E)
% where
%       E_true - essential matrix with most points in front of the camera
%       iE - the index of the true essential matrix
%       nmax - the number of points in front of the camera
%       E - several estimates of the essential matrix in a cell array
%       p - point correspondences in the first view
%       q - point correspondences in the second view
%       K1 - the intrinsic matrix of the first camera
%       K2 - the intrinsic matrix of the second camera
% 
% Author: Zhen Zhang
% Institue: Australian National University
% Last modified: 6 Jun. 2018

nE = numel(E);      % get the number of solutions
nmax = 0;            % intialize
iE = 1;

W = [0, -1, 0;...
     1, 0, 0;...
     0, 0, 1];
 
Z = [0, 1, 0;...
     -1, 0, 0;...
     0, 0, 0];

if size(p, 2) ~= 3
    error('Invalid format of points. Reformat to Nx3.');
end
 
if nE == 1
    % If there is only one solution, return it
    E_true = E{1};      
else

for i = 1: nE           % go through all E matrices
    [U, ~, V] = svd(E{i});
    for j = 1: 2        % go through all R and t
        t = U * Z * U';
        t = [t(3, 2); t(1, 3); t(2, 1)];
        switch(j)
            case 1
                R = U * W * V';
            case 2
                R = U * W' * V';
        end
        
        P1 = K1 * [eye(3), zeros(3, 1)];    % camera matrix
        P2 = K2 * [R, t];
        npoint = size(p, 1);
        X = zeros(3, npoint);               % placeholder for 3d points
        for k = 1: npoint
            % get 3d points via triangulation
            X(:, k) = tri3d(p(k, :), q(k, :), P1, P2);
        end
        % project those points back to both cameras
        p1p = P1 * [X; ones(1, npoint)];
        p2p = P2 * [X; ones(1, npoint)];
        if sum((p1p(3, :) > 0) .* (p2p(3, :) > 0)) > nmax
            % note down maximum number of points in front of both cameras
            nmax = sum((p1p(3, :) > 0) .* (p2p(3, :) > 0)); 
            % note down the index
            iE = i;        
        end
    end
end

% Return the true solution
E_true = E{iE}; 

end

end