%% PREPROCESSING STEP
%
% AUTHOR: Raghav Saboo
% DATE: 06/25/2015
%
% DESCRIPTION:
% This script uses the 100 image and prescreener data set and cleans it 
% ready for analysis.
%
% Steps included are:
% 1. Removing duplicate regions detected by the prescreener (mean shift)
% 2. Create crops of images around each of the "target" regions (10 px
%    border)
% 3. Convert regions to grayscale and normalize histogram of intensities to
%    a distribution with mean = 0 and sd = 1
% 4. Create a Gabor filter bank and normalise (L1 normalisation)
% 5. Apply Gabor filter by pixel for each region crop
% 6. Contrast normalise filter response at each pixel
% 7. Save preprocessed data for later parts

%% PRELIMINARIES
clear all; close all; clc; 
rootDirectory = 'C:/Users/Raghav/Dropbox/Solar PV Research/Old Solar PV Data';
cd(rootDirectory)
%% LOAD AND PREPARE RELEVANT DATA
% Load prescreener targets
fNameAlarms = fullfile(rootDirectory,'dsSolarPrescreener.mat');
data = load(fNameAlarms);
dsSolarPrescreener = data.dsSolarPrescreener;

% Remove NaN Values
nanIndex = find(isnan([dsSolarPrescreener.observationInfo.xI]));
dsSolarPrescreener = dsSolarPrescreener.removeObservations(nanIndex);

% Load image data
fNameImages = fullfile(rootDirectory,'roofImages.mat');
data = load(fNameImages);
imSet = data.imSet;  
imageLabels = data.tSet;

%% VISUALIZE REGIONS
imId = [dsSolarPrescreener.observationInfo.tI];
imNumbers = unique(imId);

for imUnique = 1:length(imNumbers)
    imNum = imNumbers(imUnique);
    
    imNow = imSet{imNum};
    currentFig = figure;
    set(currentFig, 'PaperPositionMode', 'auto');
    currentFig.PaperPosition = [0 0 3 4];
    %print('ScreenSizeFigure','-dpng','-r0')
    subplot(2,1,1);
    imagesc(imNow)
    title('MSER Targets on Image')
    axis off
    hold on;
    
    %Find the image from which each alarm came
    imId = [dsSolarPrescreener.observationInfo.tI];
    %Get all the alarms for image #1...that is the one we're looking at
    indsForImage1 = find(imId==imNum);
    dsIm1 = dsSolarPrescreener.retainObservations(indsForImage1);
    
    %Make a binary image: we will use this in the loop below
    binaryImage = zeros([size(imNow,1) size(imNow,2)]);
    
    %Plot some information from each alarm
    for iAlarm = 1:dsIm1.nObservations
        subplot(2,1,1);  %set the current axes to the first one
        plot(dsIm1.observationInfo(iAlarm).region)
        %This is the index of each pixel in the current region
        pixelsInAlarmRegion = dsIm1.observationInfo(iAlarm).pixelIdxList;
        %Mark these locations in the binary image
        binaryImage(pixelsInAlarmRegion)=iAlarm;
    end
    
    %plot the binary image
    subplot(2,1,2);
    imagesc(binaryImage);
    title('Mask of MSER Targets');
    axis off
    saveas(currentFig,['.\Prescreener Raw Results\','prescreener','_img_',num2str(imUnique),'.eps']);
    close all
end

%% REMOVE DUPLICATE REGIONS AND VISUALIZE
dsSolarWithoutDuplicates = dsSolarPrescreener;
imId = [dsSolarPrescreener.observationInfo.tI];
imNumbers = unique(imId);
regionRemoved = [];
for imUnique = 1:length(imNumbers)
    imNum = imNumbers(imUnique);
    imageIndex = find(imId == imNum);
    dsCurrentImage = dsSolarPrescreener.retainObservations(imageIndex);
    totalRegions = dsCurrentImage.nObservations;
    x = zeros(1,totalRegions);
    y = zeros(1,totalRegions);
    for regionNum = 1:totalRegions
        x(regionNum) = dsCurrentImage.observationInfo(regionNum).xI;
        y(regionNum) = dsCurrentImage.observationInfo(regionNum).yI;
    end
    points = [x; y];
    var = 1; bandwidth = 15; clustMed = [];
    [clustCent, point2cluster, clustMembsCell] = MeanShiftCluster(points,bandwidth);
    numClust = length(clustMembsCell);
        currentFig = figure(imNum);
        set(currentFig, 'PaperPositionMode', 'auto');
        currentFig.PaperPosition = [0 0 3 4];
        cVec = 'bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk';
        imagesc(imSet{imNum})
        axis off
        hold on
        for k = 1:min(numClust,length(cVec))
            myMembers = clustMembsCell{k};
            myClustCen = clustCent(:,k);
            plot(points(1,myMembers),points(2,myMembers),[cVec(k) '.'])
            plot(myClustCen(1),myClustCen(2),'o','MarkerEdgeColor','k','MarkerFaceColor',cVec(k), 'MarkerSize',10)
        end
        title(['Unique MSER Targets:' int2str(numClust)])
        saveas(currentFig,['.\Prescreener Mean Shift Results\','prescreener','_img_',num2str(imUnique),'.eps']);
        close all
    for cluster = 1:numClust
        regionIndex = find(point2cluster == cluster);
        numRegions = length(regionIndex);
        if numRegions == 1
            continue
        else
            for rI = 1:numRegions
                numPixels(rI) = length(dsCurrentImage.observationInfo(regionIndex(rI)).pixelList);
            end
            maxPixels = find(max(numPixels));
            regionIndex(maxPixels) = [];
            regionIndex = regionIndex + imageIndex(1) - 1;
            regionRemoved = [regionRemoved regionIndex];
        end
    end
    %disp(imNum);
end
%dsSolarWithoutDuplicates = dsSolarPrescreener;
dsSolarWithoutDuplicates = dsSolarWithoutDuplicates.removeObservations(regionRemoved);
save('dsSolarWD','dsSolarWithoutDuplicates')
%% CREATE CROPS AROUND IMAGES
close all
imId = [dsSolarWithoutDuplicates.observationInfo.tI];
imNumbers = unique(imId);
for imUnique = 1:length(imNumbers)
    imNum = imNumbers(imUnique);
    imNow = imSet{imNum};
    imSize = size(imNow);
    % Figure placeholder
    %     figure(imNum);
    %Get all the alarms for the current image
    indsForImage1 = find(imId==imNum);
    dsIm1 = dsSolarWithoutDuplicates.retainObservations(indsForImage1);
    %Make a binary image: we will use this in the loop below
    binaryImage = zeros([size(imNow,1) size(imNow,2)]);
    numRegions=dsIm1.nObservations; % number of objects
    %Plot some information from each alarm
    %     subplot(1,2,1);
    %     imagesc(imNow);
    %     title('Original image with new regions marked')
    %     hold on;
    for iAlarm = 1:numRegions
        %          subplot(1,2,1);  %set the current axes to the first one
        %          plot(dsIm1.observationInfo(iAlarm).region)
        %This is the index of each pixel in the current region
        pixelsInAlarmRegion = dsIm1.observationInfo(iAlarm).pixelIdxList;
        %Mark these locations in the binary image
        binaryImage(pixelsInAlarmRegion)=iAlarm;
    end
    % Get the bounding box around each object
    regionBoundingBox=regionprops(binaryImage,'BoundingBox');
    % Crop the individual objects and store them in a cell
    regionCell=cell(numRegions,1);
    % Pixel offsets in the x and y directions
    xOffset = 1; yOffset = 1;
    for i=1:numRegions
        % Get the bounding box of the i-th region and offest by x and y
        % pixels in their respective directions
        rbb_i=ceil(regionBoundingBox(i).BoundingBox);
        idx_x=[rbb_i(1)-xOffset rbb_i(1)+rbb_i(3)+xOffset-1];
        idx_y=[rbb_i(2)-yOffset rbb_i(2)+rbb_i(4)+yOffset-1];
        if idx_x(1)<1, idx_x(1)=1; end
        if idx_y(1)<1, idx_y(1)=1; end
        if idx_x(2)>imSize(2), idx_x(2)=imSize(2); end
        if idx_y(2)>imSize(1), idx_y(2)=imSize(1); end
        % Crop the object and write to ObjCell
        im=binaryImage==i;
        dsSolarWithoutDuplicates.observationInfo(indsForImage1(i)).croppedRegionBinary = im(idx_y(1):idx_y(2),idx_x(1):idx_x(2));
        dsSolarWithoutDuplicates.observationInfo(indsForImage1(i)).regionRGB = imNow.*uint8(repmat(im,[1,1,3]));
        dsSolarWithoutDuplicates.observationInfo(indsForImage1(i)).croppedRegionRGB = dsSolarWithoutDuplicates.observationInfo(indsForImage1(i)).regionRGB(idx_y(1):idx_y(2),idx_x(1):idx_x(2),:);
        dsSolarWithoutDuplicates.observationInfo(indsForImage1(i)).croppedRGB = imNow(idx_y(1):idx_y(2),idx_x(1):idx_x(2),:);
        dsSolarWithoutDuplicates.observationInfo(indsForImage1(i)).croppedGrayscale = rgb2gray(dsSolarWithoutDuplicates.observationInfo(indsForImage1(i)).croppedRGB);
    end
    %Visualize the individual objects
%     figure(imUnique)
%         for i=1:numRegions
%             subplot(1,numRegions,i)
%             imshow(dsSolarWithoutDuplicates.observationInfo(i).croppedRegionRGB)
%         end
end
save('dsSolarWDC','dsSolarWithoutDuplicates')
%% NORMALIZE INTENSITY OF GRAYSCALE CROPS
imId = [dsSolarWithoutDuplicates.observationInfo.tI];
imNumbers = unique(imId);
figure
for i=1:36
        %figure
        %imagesc(dsSolarWithoutDuplicates.observationInfo(imUnique).croppedGrayscale);
        dsSolarWithoutDuplicates.observationInfo(i).croppedGrayscaleNormalized = histeq(dsSolarWithoutDuplicates.observationInfo(i).croppedGrayscale, imhist(dsSolarWithoutDuplicates.observationInfo(1).croppedGrayscale));
        %pause(1)
        subplot(6,6,i)
        imagesc(dsSolarWithoutDuplicates.observationInfo(i).croppedGrayscaleNormalized)
        axis off
end
%save('dsSolarWDCGN','dsSolarWithoutDuplicates')
%% Convex Hull Rotation
%Plot covex hull
numRegions = dsSolarWithoutDuplicates.nObservations;
figure
for regnum = 1:36 %numRegions
    croppedBinary = dsSolarWithoutDuplicates.observationInfo(regnum).croppedRegionBinary;
    croppedBinary = bwconvhull(croppedBinary);
    subplot(6,6,regnum)
    imshow(croppedBinary)
    title([num2str(dsSolarWithoutDuplicates.targets(regnum))])
    hold on
end
% Plot convex hull rotated to normalise orientation
figure
for regnum = 1:36
    croppedBinary = dsSolarWithoutDuplicates.observationInfo(regnum).croppedRegionBinary;
    grayscaleRegion = dsSolarWithoutDuplicates.observationInfo(regnum).croppedGrayscaleNormalized;
    if hasInfNaN(grayscaleRegion) || hasInfNaN(croppedBinary) || length(grayscaleRegion) <= 1 || length(croppedBinary) <= 1
        continue
    else
        croppedBinary = bwconvhull(croppedBinary);
        measurement = regionprops(croppedBinary, 'Orientation');
        angle = mean([measurement.Orientation]);
        croppedBinary = imrotate(croppedBinary,-angle,'nearest');
        croppedGrayscaleNormalized = imrotate(grayscaleRegion,-angle,'nearest');
        dsSolarWithoutDuplicates.observationInfo(regnum).croppedGrayscaleNormalized = croppedGrayscaleNormalized;
        extentRatio=regionprops(croppedBinary,'Extent');
        extentRatio=[extentRatio.Extent];
        subplot(6,6,regnum)
        imshow(croppedBinary)
        %title([num2str(dsSolarWithoutDuplicates.targets(regnum)),' & ',num2str(extentRatio)])
        axis off
        hold on
    end
end
%% GABOR FILTERS FEATURES ARE AUTOMATICALLY NORMALISED BY FUNCTION
close all
gaborArray = gaborFilterBank(1,8,15,15);
numRegions = dsSolarWithoutDuplicates.nObservations;
gaborFeatMatrix = zeros(numRegions,1);
targets = dsSolarWithoutDuplicates.targets;

for regionNum = 1:numRegions
    grayscaleRegion = dsSolarWithoutDuplicates.observationInfo(regionNum).croppedGrayscaleNormalized;
    [u,v] = size(gaborArray);
    gaborResult = cell(u,v);
    [n,m] = size(grayscaleRegion);
    s = (n*m); l = s*u*v;
    for i = 1:u
        for j = 1:v
            gaborResult{i,j} = conv2(double(grayscaleRegion), double(gaborArray{i,j}), 'same');
        end            
    end
    dsSolarWithoutDuplicates.observationInfo(regionNum).gaborResult = gaborResult;
end
% Show magnitudes of Gabor-filtered images
cd([rootDirectory,'\Gabor Filter Results (1,8,3,3)\'])
for regnum = 1:numRegions
    h = figure('Visible','off');
    axes('position', [0 0 1 1])
    set(h, 'PaperPositionMode', 'auto');
    h.PaperPosition = [0 0 6 2];
    regionNum = regnum;
    for i = 1:u
        for j = 1:v
            subplot(u,v,(i-1)*v+j)
            imshow(abs(dsSolarWithoutDuplicates.observationInfo(regionNum).gaborResult{i,j}),[])
        end
    end
    suptitle(['Is Solar PV: ',num2str(dsSolarWithoutDuplicates.targets(regnum))])
    set(h, 'Visible', 'off')
    saveas(h,[num2str(dsSolarWithoutDuplicates.targets(regnum)),'fig',num2str(regionNum),'.png']);
end
% Show real parts of Gabor-filtered images
%figure('NumberTitle','Off','Name','Real parts of Gabor filters');
%regionNum = 2;
%for i = 1:u
%    for j = 1:v
%        subplot(u,v,(i-1)*v+j)
%        imshow(real(dsSolarWithoutDuplicates.observationInfo(regionNum).gaborResult{i,j}),[]);
%    end
%end
cd([rootDirectory]);
save('dsSolarPreprocessed1515','dsSolarWithoutDuplicates')
%% SCORE THE PRESCREENER CONFIDENCE WITHOUT ANY CLASSIFICATION
figure;  %Create a figure
%Plot the ROC curve
[pf,pd] = prtScoreRoc(dsSolarWithoutDuplicates);
%Get AUC and add it in legend
myAuc = prtScoreAuc(dsSolarWithoutDuplicates);
plot(pf,pd,'r');
legendStr = sprintf('Prescreener, AUC=%0.3f',myAuc);
xlabel('Pf');
ylabel('Pd');
title('Performance of the prescreener');

%This places the number of false alarms on the x-axis rather than the
figure;
% proportion of false alarms
[nfa,pd] = prtScoreRocNfa(dsSolarWithoutDuplicates);
plot(nfa,pd);
xlabel('number of false alarms');
ylabel('proportion of panels found');
title('Illustration of plotting ROC with no. of false alarms');

%% ADD CLASSIFIER
algo = prtOutlierRemovalMissingData+prtPreProcZmuv+prtClassLibSvm;