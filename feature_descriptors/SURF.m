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