%% TEXTON CLUSTERING
% 
% AUTHOR: Raghav Saboo
% DATE: 11/04/2015
% 
% DESCRIPTION:
% This code will cluster the filter responses using k-Means, k- clustering and
% generate a texton dictionary. This texton dictionary is used to then
% define texture models based on texton frequencies learnt from training
% images.

%% PRELIMINARIES
clc; clear all; close all;
rootDirectory = 'C:/Users/Raghav/Dropbox/Solar PV Research/Old Solar PV Data';
cd(rootDirectory)
%% LOAD RELEVANT DATA
% Load pre-processed data
fileName = fullfile(rootDirectory,'dsSolarPreprocessed.mat');
data = load(fileName);
dsSolarPreprocessed = data.dsSolarWithoutDuplicates;
numRegions = dsSolarPreprocessed.nObservations;

% Load image data
fNameImages = fullfile(rootDirectory,'roofImages.mat');
data = load(fNameImages);
imSet = data.imSet;  
imageLabels = data.tSet;

%% SECTION THE DATA
% Solar PV Regions
numSamples = 10;
[sample,indices] = bootstrapByClass(dsSolarPreprocessed,numSamples);
solarPVSample = retainClasses(sample,1);
nonPVSample = retainClasses(sample,0);
testData = removeObservations(dsSolarPreprocessed,indices);
%% HISTOGRAMS OF GABOR FILTER RESULTS
close all
gaborArray = gaborFilterBank(1,8,15,15);
cd([rootDirectory,'\Gabor Filter Results (1,8,3,3)\Sample Histograms'])
edges = [0:0.1:10];
for regnum = 1:10
    [n,m] = size(solarPVSample.observationInfo(regnum).gaborResult{1,1});
    h = figure('Visible','off');
    histogram(real(solarPVSample.observationInfo(regnum).gaborResult{1,1}),edges,'Normalization','probability')
    title(['Gabor Filter {1,1} Hist. Is Solar PV: 1'])
    saveas(h,['PV_','fig',num2str(regnum),'.png']);
    solarPVSample.observationInfo(regnum).histcounts = histcounts(abs(solarPVSample.observationInfo(regnum).gaborResult{1,1}),edges);
    h = figure('Visible','off');
    histogram(real(nonPVSample.observationInfo(regnum).gaborResult{1,1}),edges,'Normalization','probability')
    title(['Gabor Filter {1,1} Hist. Is Solar PV: 0'])
    saveas(h,['nonPV_','fig',num2str(regnum),'.png']);
    nonPVSample.observationInfo(regnum).histcounts = histcounts(abs(solarPVSample.observationInfo(regnum).gaborResult{1,1}),edges);
    %figure
    %imagesc(abs(solarPVSample.observationInfo(regnum).gaborResult{1,1}))
end
%% PHASE AND MAGNITUDE CALCULATIONS OF GABOR FILTER RESULTS
solarPVcomplex = [];
nonPVcomplex = [];
%close all
for regnum = 1:numSamples
    for filter = 1:8
        [n,m] = size(solarPVSample.observationInfo(regnum).gaborResult{1,filter});
        angles = reshape(angle(solarPVSample.observationInfo(regnum).gaborResult{1,filter}),[n*m,1]);
        magnitude = reshape(abs(solarPVSample.observationInfo(regnum).gaborResult{1,filter}),[n*m,1]);
        solarPVcomplex = [solarPVcomplex; [angles magnitude]];
    end
end
figure
plot(solarPVcomplex(:,1),solarPVcomplex(:,2),'.','markers',0.01)
figure
polar(solarPVcomplex(:,1),solarPVcomplex(:,2),'.')
for regnum = 1:numSamples
    for filter = 1:8
        [n,m] = size(nonPVSample.observationInfo(regnum).gaborResult{1,filter});
        angles = reshape(angle(nonPVSample.observationInfo(regnum).gaborResult{1,filter}),[n*m,1]);
        magnitude = reshape(abs(nonPVSample.observationInfo(regnum).gaborResult{1,filter}),[n*m,1]);
        nonPVcomplex = [nonPVcomplex; [angles magnitude]];
    end
end
figure
plot(nonPVcomplex(:,1),nonPVcomplex(:,2),'.','markers',0.01)
figure
polar(nonPVcomplex(:,1),nonPVcomplex(:,2),'.')
%% EXTRACT HOG FEATURES
solarPVhog = [];
nonPVhog = [];
for regnum = 1:numSamples
    for filter = 1:8
        [n,m] = size(solarPVSample.observationInfo(regnum).gaborResult{1,filter});
        angles = reshape(angle(solarPVSample.observationInfo(regnum).gaborResult{1,filter}),[n*m,1]);
        magnitude = reshape(abs(solarPVSample.observationInfo(regnum).gaborResult{1,filter}),[n*m,1]);
        solarPVcomplex = [solarPVcomplex; [angles magnitude]];
    end
end
%figure
%plot(solarPVcomplex(:,1),solarPVcomplex(:,2),'.')
%figure
%polar(solarPVcomplex(:,1),solarPVcomplex(:,2),'.')
for regnum = 1:numSamples
    for filter = 1:8
        [n,m] = size(nonPVSample.observationInfo(regnum).gaborResult{1,filter});
        angles = reshape(angle(nonPVSample.observationInfo(regnum).gaborResult{1,filter}),[n*m,1]);
        magnitude = reshape(abs(nonPVSample.observationInfo(regnum).gaborResult{1,filter}),[n*m,1]);
        nonPVcomplex = [nonPVcomplex; [angles magnitude]];
    end
end
%% (NOT CHOSEN) USING PRT TOOLBOX (REALLY SLOW)
figure
solarPVds = prtDataSetClass(solarPVcomplex);
clusterAlgo = prtClusterKmodes;
clusterAlgo.nClusters = 6;
clusterAlgo.internalDecider = prtDecisionMap;
clusterAlgo = clusterAlgo.train(solarPVds);
figure
plot(clusterAlgo);

figure
nonPVds = prtDataSetClass(nonPVcomplex);
clusterAlgo1 = prtClusterKmodes;
clusterAlgo1.nClusters = 6;
clusterAlgo1.internalDecider = prtDecisionMap;
clusterAlgo1 = clusterAlgo1.train(nonPVds);
figure
plot(clusterAlgo1);
%% (CHOSEN) K-MEDIODS USING STATISTICAL TOOLBOX IN MATLAB
%close all
rng('default')
clusters = 4;
[idpv,Cpv,sumDpv,Dpv] = kmedoids(solarPVcomplex,clusters);
solarPVcomplex(:,3) = zeros(length(idpv),1);
for k = 1:clusters
    for i = 1:length(idpv)
        if idpv(i,1) == k
            solarPVcomplex(i,3) = Dpv(i,k);
        end
    end
end
figure
for k = 1:clusters
plot(solarPVcomplex(idpv==k,1),solarPVcomplex(idpv==k,2),'.','MarkerSize',0.01);
hold on
end
title('PV Data K-Mediods Clusters')
hold off
[idnonpv,Cnonpv,sumDnonpv,Dnonpv] = kmedoids(nonPVcomplex,clusters);
nonPVcomplex(:,3) = zeros(length(idnonpv),1);
for k = 1:clusters
    for i = 1:length(idnonpv)
        if idnonpv(i,1) == k
            nonPVcomplex(i,3) = Dnonpv(i,k);
        end
    end
end
figure
for k = 1:clusters
plot(nonPVcomplex(idnonpv==k,1),nonPVcomplex(idnonpv==k,2),'.','MarkerSize',0.01);
hold on
end
title('Non PV Data K-Mediods Clusters')
hold off
figure
scatter3(nonPVcomplex(:,1),nonPVcomplex(:,2),nonPVcomplex(:,3),'.');
hold on
scatter3(solarPVcomplex(:,1),solarPVcomplex(:,2),solarPVcomplex(:,3),'.');

edges = [0:0.5:70];
figure
histogram(Dpv,edges,'Normalization','probability');
[p1,n1] = histcounts(Dpv,edges,'Normalization','probability');
figure
histogram(Dnonpv,edges,'Normalization','probability');
[p2,n2] = histcounts(Dnonpv,edges,'Normalization','probability');
d = pdist2(p1,p2);
figure
histogram(idpv)
figure
histogram(idnonpv)
solarPVcomplex(:,3) = []; nonPVcomplex(:,3) = [];

%% FIT GMM DIST
rng default
clusters = 4;
options = statset('Display','final');
GMModelPV = fitgmdist(solarPVcomplex,clusters,'Options',options,'Start',idpv);
GMModelnonPV = fitgmdist(nonPVcomplex,clusters,'Options',options,'Start',idnonpv);
figure
h = gscatter(nonPVcomplex(:,1),nonPVcomplex(:,2),idnonpv,[],'.')
haxis = gca;
xlim = haxis.XLim;
ylim = haxis.YLim;
d = (max([xlim ylim])-min([xlim ylim]))/1000;
[X1Grid,X2Grid] = meshgrid(xlim(1):d:xlim(2),ylim(1):d:ylim(2));
%hold on
%contour(X1Grid,X2Grid,reshape(pdf(GMModelnonPV,[X1Grid(:) X2Grid(:)]),...
%    size(X1Grid,1),size(X1Grid,2)),20)
%uistack(h,'top')
%legend off
%title('Non PV Data K-Mediods Clusters')
%hold off
disp(GMModelPV)
disp(GMModelnonPV)
%% (NOT CHOSEN) K-MEANS USING STATISTICAL TOOLBOX IN MATLAB
close all
rng('default')
clusters = 4;
[idpv,Cpv,sumDpv,Dpv] = kmeans(solarPVcomplex,clusters);
figure
for k = 1:clusters
plot(solarPVcomplex(idpv==k,1),solarPVcomplex(idpv==k,2),'.');
hold on
end
title('PV Data K-Means Clusters')
hold off
[idnonpv,Cnonpv,sumDnonpv,Dnonpv] = kmeans(nonPVcomplex,clusters);
figure
for k = 1:clusters
plot(nonPVcomplex(idnonpv==k,1),nonPVcomplex(idnonpv==k,2),'.');
hold on
end
title('Non PV Data K-Means Clusters')
hold off
edges = [0:0.5:70];
figure
histogram(Dpv,edges,'Normalization','probability');
[p1,n1] = histcounts(Dpv,edges,'Normalization','probability');
figure
histogram(Dnonpv,edges,'Normalization','probability');
[p2,n2] = histcounts(Dnonpv,edges,'Normalization','probability');
d = pdist2(p1,p2);
%% (NOT CHOSEN) GAUSSIAN MIXTURE MODEL 
close all
rng('default')
components = 3;
%figure
%scatter(solarPVcomplex(:,1),solarPVcomplex(:,2),0.5,'.')
obj = fitgmdist(solarPVcomplex,components);
P = posterior(obj,solarPVcomplex);
figure
for i = 1:components
    subplot(3,1,i);
    scatter(solarPVcomplex(:,1),solarPVcomplex(:,2),0.5,P(:,i),'.')
    hb = colorbar;
    ylabel(hb,['Component ',num2str(i),' Probability'])
end

%figure
%scatter(nonPVcomplex(:,1),nonPVcomplex(:,2),0.5,'.')
%obj = fitgmdist(nonPVcomplex,components);
P = posterior(obj,nonPVcomplex);
figure
for i = 1:components
    subplot(3,1,i);
    scatter(nonPVcomplex(:,1),nonPVcomplex(:,2),0.5,P(:,i),'.')
    hb = colorbar;
    ylabel(hb,['Component ',num2str(i),' Probability'])
end
%% CALCULATION FOR TEST DATA SET
testComplex = [];
numRegions = testData.nObservations;
for regnum = 1:numRegions
    testComplex = [];
    for filter = 1:8
        [n,m] = size(testData.observationInfo(regnum).gaborResult{1,filter});
        angles = reshape(angle(testData.observationInfo(regnum).gaborResult{1,filter}),[n*m,1]);
        magnitude = reshape(abs(testData.observationInfo(regnum).gaborResult{1,filter}),[n*m,1]);
        testComplex = [testComplex; [angles magnitude]];
    end
    testData.observationInfo(regnum).testComplex = testComplex;
end
%figure
%plot(solarPVcomplex(:,1),solarPVcomplex(:,2),'.')
%figure
%polar(solarPVcomplex(:,1),solarPVcomplex(:,2),'.')
%% COMPARISON DISTANCE USING K-MEDIODS
%close all
rng('default')
clusters = 4;
edges = [0:0.5:70];
for regnum = 1:numRegions
    [idpv,Cpv,sumDpv,Dpv] = kmedoids(testData.observationInfo(regnum).testComplex,clusters);
    [ptest,ntest] = histcounts(Dpv,edges,'Normalization','probability');
    testData.observationInfo(regnum).HistPVdist = pdist2(ptest,p1);
    testData.observationInfo(regnum).HistNonPVdist = pdist2(ptest,p2);
    testData.observationInfo(regnum).Dpv = Dpv;
    testData.observationInfo(regnum).Cpv = Cpv;
    testData.observationInfo(regnum).sumDpv = sumDpv;
end
%% CREATE FEATURES FROM K-MEDIODS
%testData.data(:,2:5) = [];
for regnum = 1:numRegions
    testData.data(regnum,2) = testData.observationInfo(regnum).HistPVdist;
    testData.data(regnum,3) = testData.observationInfo(regnum).HistNonPVdist;
    testData.data(regnum,4) = max(max(testData.observationInfo(regnum).Dpv));
end
%% GMM PROBABILITY FEATURE
for regnum = 1:numRegions
    testComplex = testData.observationInfo(regnum).testComplex;
    [probPV,logLpv] = posterior(GMModelPV,testComplex); 
    testData.observationInfo(regnum).pPV = mean(quantile(probPV,0.90));
    testData.observationInfo(regnum).logLpv = logLpv;
    [probNonPV,logLnonPV] = posterior(GMModelnonPV,testComplex);
    testData.observationInfo(regnum).pNonPV = mean(quantile(probNonPV,0.90));
    testData.observationInfo(regnum).logLnonPV = logLnonPV;
    testData.data(regnum,5) = testData.observationInfo(regnum).pPV;
    testData.data(regnum,6) = testData.observationInfo(regnum).pNonPV;
end
%% VISUALISATION OF SHAPE
figure
for regnum = 1:36
    croppedBinary = testData.observationInfo(regnum).croppedRegionBinary;
    subplot(6,6,regnum)
    imshow(croppedBinary)
    title([num2str(testData.targets(regnum))])
    hold on
end
% Plot covex hull
figure
for regnum = 1:36
    croppedBinary = testData.observationInfo(regnum).croppedRegionBinary;
    croppedBinary = bwconvhull(croppedBinary);
    subplot(6,6,regnum)
    imshow(croppedBinary)
    title([num2str(testData.targets(regnum))])
    hold on
end
% Plot convex hull rotated to normalise orientation
figure
for regnum = 1:36
    croppedBinary = testData.observationInfo(regnum).croppedRegionBinary;
    croppedBinary = bwconvhull(croppedBinary);
    measurement = regionprops(croppedBinary, 'Orientation');
    angle = mean([measurement.Orientation]);
    croppedBinary = imrotate(croppedBinary,-angle,'nearest');
    extentRatio=regionprops(croppedBinary,'Extent');
    extentRatio=[extentRatio.Extent];
    subplot(6,6,regnum)
    imshow(croppedBinary)
    title([num2str(testData.targets(regnum)),' & ',num2str(extentRatio)])
    hold on
end
%for regnum = 1:numRegions
%    if testData.observationInfo(regnum).HistPVdist < testData.observationInfo(regnum).HistNonPVdist
%        testData.data(regnum,2) = 1;
%    else
%        testData.data(regnum,2) = 0;
%    end
%end
%% ADD FEATURE OF 'EXTENT'
for regnum = 1:numRegions
    croppedBinary = testData.observationInfo(regnum).croppedRegionBinary;
    [n,m] = size(croppedBinary);
    if n == 1 && m == 1
        extentRatio = 0;
        solidityRatio = 0;
    else
        solidityRatio = regionprops(croppedBinary,'Solidity');
        solidityRatio=[solidityRatio.Solidity];
        solidityRatio = mean(solidityRatio);
        croppedBinary = bwconvhull(croppedBinary);
        measurement = regionprops(croppedBinary, 'Orientation');
        angle = mean([measurement.Orientation]);
        croppedBinary = imrotate(croppedBinary,-angle,'nearest');
        extentRatio=regionprops(croppedBinary,'Extent');
        extentRatio=[extentRatio.Extent];
    end
    testData.observationInfo(regnum).extent = extentRatio;
    testData.data(regnum,7) = testData.observationInfo(regnum).extent;
    testData.observationInfo(regnum).solidity = solidityRatio;
    testData.data(regnum,8) = testData.observationInfo(regnum).solidity;
end
%% CONTOUR CURVATURE
%close all
cd([rootDirectory]);
for regnum = 11:36
    croppedBinary = testData.observationInfo(regnum).croppedRegionBinary;
    regionRGB = testData.observationInfo(regnum).regionRGB;
    [n,m] = size(croppedBinary);
    if n == 0 && m == 0 || n == 1 && m == 1
        n = 0;
        bin = 0;
        hyp = 1;
        pval = 1;
        n90 = 0;
        n0 = 0;
    else
        edgeBinary = edge2(croppedBinary);
        edgeBinary = padarray(edgeBinary,[10,10]);
        figure
        imshow(~edgeBinary);
        Orientations = skeletonOrientation(edgeBinary,5);
        Onormal = Orientations+90; %easier to view normals
        Onr = sind(Onormal); %vv
        Onc = cosd(Onormal); %uu
        [r,c] = find(edgeBinary);    %row/cols
        idx = find(edgeBinary);      %Linear indices into Onr/Onc
        %Overlay normals to verify
        hold on
        quiver(c,r,-Onc(idx),Onr(idx));
        listOrient = Orientations(edgeBinary);
        figure
        edges = [-180:1:180];
        histogram(listOrient,edges,'Normalization','probability')
        [n,bin] = histcounts(listOrient,edges,'Normalization','probability');
        [hyp,pval] = chi2gof(listOrient);
        n90 = n(bin == 90) + n(bin == -90);
        n0 = n(bin == 0);
    end
    testData.observationInfo(regnum).orientCount = n;
    testData.observationInfo(regnum).orientAngle = bin;
    testData.observationInfo(regnum).orientNormal = ~hyp;
    testData.observationInfo(regnum).orientPval = pval;
    testData.observationInfo(regnum).orient90 = n90;
    testData.observationInfo(regnum).orient0 = n0;
    testData.data(regnum,9) = testData.observationInfo(regnum).orientNormal;
    testData.data(regnum,10) =  testData.observationInfo(regnum).orientPval;
    %testData.data(regnum,11) =  testData.observationInfo(regnum).orient0;
    %testData.data(regnum,12) =  testData.observationInfo(regnum).orient90;
    testData.data(regnum,12) = mean(mean(regionRGB(:,:,1)));
    testData.data(regnum,13) = mean(mean(regionRGB(:,:,2)));
    testData.data(regnum,14) = mean(mean(regionRGB(:,:,3)));
    testData.data(regnum,11) = mean(testData.data(regnum,12:14));
end
%% FEATURE MAP
plot(testData);
%% PCA Map
pca = prtPreProcPca;            % Create a prtPreProcPca object
                         
    pca = pca.train(testData);       % Train the prtPreProcPca object
    dataSetNew = pca.run(testData);  % Run
  
    % Plot
    figure
    plot(dataSetNew);
    title('PCA Projected Data');
    
    pls = prtPreProcPls;                   % Create a prtPreProcPls Object
                         
    pls = pls.train(testData);              % Train
    dataSetNew = pls.run(testData);         % Run
 
    % Plot 
    figure
    plot(dataSetNew);
    title('PLS Projected Data');
    
    lda = prtPreProcLda;                    % Create the pre-processor
 
    lda = lda.train(testData);               % Train
    dataSetNew = lda.run(testData);          % Run
 
    % Plot the results
    figure
    plot(dataSetNew);
    title('LDA Projected Data');
%% RESULTS
kfolds = 100;
svm = prtClassLibSvm;
svm.kernelType = 2;
svm.gamma = 1;
svm.cost = 1;
svm.shrinking = 1;
algoSVM = prtOutlierRemovalMissingData + prtPreProcZmuv + svm;
[svmBasicFeaturesOut,algoTrained] = algoSVM.kfolds(testData,kfolds);
figure
prtScoreRoc(svmBasicFeaturesOut);
figure
prtScoreRocNfa(svmBasicFeaturesOut);