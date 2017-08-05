% clear
clear;
close all;

% code
addpath('../../src/frame');
addpath('../../src/image');

% config
para = config();

% TODO: override, fix this
para.dataPath = 'dataset/rotate/';
para.categoryName = 'apple_test';
para.name = 'rotate_test';

task_id = 1;
noWorkers = 1;

% mex
compileMex('../../src/frame');

% set seed for random numbers generation
seed=1;
rng(seed);

% EM clustering parameters
numEMIteration = para.numEMIteration;
numCluster = para.numCluster;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% parameters for two-stage learning algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigsq = 10;
numSketch = para.numWavelet; %400 sparsification
numTopFeatureToShow = 70; % not used so far, please ignore
lambdaLearningRate_MP = 0.1/sqrt(sigsq);  % 0.1
epsilon = 0.03;
L = 10;
nIter = 80;
numSample = 3; % how many HMC calls for each learning iteration
isSaved = 1;

%%%%%%% parameters for multiple selection (cell size) or deformable parts %%%%%%%%%%%%%%%%%%%
nPartCol = para.nPartCol;
nPartRow = para.nPartRow;
part_sx = para.part_sx;
part_sy = para.part_sy;

gradient_threshold_scale = 0.8; %  we use adaptive threshold in the multiple selection. The threshold is equal to (gradient_threshold_scale * maximum gradient).
numPart = nPartCol*nPartRow; % number of deformable parts

argmaxMethod = 1;  % 1: local max in squared region, 0: cross-shaped rgion for parts
relativePartRotationRange = para.relativePartRotationRange;  % relative rotation range for parts
relativePartLocationRange= para.relativePartLocationRange; % default 1
resolutionShiftLimit = para.resolutionShiftLimit;
%%%%%%%

%%%%%%% parameters for large template
sx = nPartRow*part_sx;   % template size x
sy = nPartCol*part_sy;   % template size y
rotateShiftLimit = para.rotateShiftLimit;   % template rotation from -rotateShiftLimit to rotateShiftLimit, eg. (1)-2:2 if rotateShiftLimit=2 (2)0 is without rotation
rotationRange = -rotateShiftLimit:rotateShiftLimit;
numRotate = length(rotationRange);
RatioDisplacementSUM3 = para.ratioDisplacementSUM3;   % default=0. Compute all values in SUM3 map

nOrient = para.nOrient;
locationShiftLimit = para.locationShiftLimit;  % arg-max
orientShiftLimit = para.orientShiftLimit;    % arg-max
%%%%%%%

%%%%%%% parameters about resolution
numResolution = para.numResolution;  % number of resolutions to search for in detection stage, the middle level is the original size
scaleStepSize = 0.1; % you can tune this either 0.1 or 0.2
originalResolution = round(numResolution/2);    % original resolution is the one at which the imresize factor = 1.  the middle one is by default
displace = 0;
originalResolution = originalResolution - displace;    % shift the index of the original resolution to the left.  We don't want too many shrinkage resolution.
%%%%%%%

%%%%%%%
interval = 1; % 5 control how many iterations we need to wait to select next wavlet
numWavelet = para.numWavelet;   % default 300, 370
%%%%%%%

%%%%%%% parameters for Gibbs sampler
threshold_corrBB = 0;    % threshold for correlation matrix  0.01
lower_bound_rand = 0.001;   % lower bound of random number
upper_bound_rand = 0.999;   % upper bound of random number
c_val_list=-25:3:25;  % range of c
lambdaLearningRate_boosting = 0.1/sqrt(sigsq);  % 0.1
%%%%%%%

%%%%%%%%%%%%%% multiple chains
useMultiChain = true;
nTileRow = 10; %nTileRow \times nTileCol defines the number of paralle chains
nTileCol = 10;
if useMultiChain == false
    nTileRow = 1;
    nTileCol = 1;
end
%%%%%%%%%%%%%

%%% local normalization parameters
isLocalNormalize = para.isLocalNormalize; %false; %
isSeparate = false;
localNormScaleFactor = 2; %0.5, 1, 2, 3, or 4, we suggest 2;
thresholdFactor = 0.01;
%%%%%%%%%%%%%%%%%%%%%%%%%%

inPath = [para.dataPath '/' para.categoryName];
cachePath = ['./output/' 'rotate' '/feature'];
templatePath = ['./output/' 'rotate' '/template'];
resultPath = ['./output/' para.name '/result_seed_' num2str(seed)];

if exist(['./output/' para.name],'dir')
    rmdir(['./output/' para.name],'s')
end

if ~exist(resultPath,'dir')
    mkdir(resultPath)
    mkdir(fullfile(resultPath,'img'));
end

%% Step 0: prepare filter, training images, and filter response on images
GaborScaleList = para.GaborScaleList;
DoGScaleList = para.DoGScaleList;

if isLocalNormalize
    DoGScaleList=[];
end

nScaleDoG=length(DoGScaleList);
nScaleGabor=length(GaborScaleList);

%%%%%%%%%%%%%%% auxiliary variables for hierachical template
partRotations=cell(numRotate, 1);
partRotationRange = [];   % save the possible range of part rotation
for ii = 1:numRotate
    tmpRotation = rotationRange(ii)+relativePartRotationRange;
    partRotations{ii} = tmpRotation;
    partRotationRange = union( partRotationRange, tmpRotation );
end
numPartRotate = length(partRotationRange);

% record the top-left location of each part in a large template
PartLocX0 = 1:part_sx:sx-part_sx+1;
PartLocY0 = 1:part_sy:sy-part_sy+1;

% another representation of the top-left location
PartLocX = zeros(numPart,1);
PartLocY = zeros(numPart,1);
iPart = 1;
for x = PartLocX0
    for y = PartLocY0
        PartLocX(iPart) = x;
        PartLocY(iPart) = y;
        iPart = iPart + 1;
    end
end

% compute affinity matrix
minRotationDif = (sin(1*pi/nOrient)-sin(0))^2 + (cos(1*pi/nOrient)-cos(0))^2 + 1e-10;

allTemplateAffinityMatrix = cell(numPartRotate, numPart);
templateAffinityMatrix = cell(numPartRotate,numPart);
for iPart = 1:numPart
    for r1 = 1:length(partRotationRange)
        angle1 = pi/nOrient * partRotationRange(r1);
        templateAffinityMatrix{r1,iPart} = [];
        jPart = iPart;
        for r2 = 1:length(partRotationRange)
            angle2 = pi/nOrient*partRotationRange(r2);
            if (sin(angle1) - sin(angle2))^2 + (cos(angle1)-cos(angle2))^2 <= minRotationDif
                templateAffinityMatrix{r1,iPart} = int32( [templateAffinityMatrix{r1,iPart} r2-1] );
            end
        end
    end
end
for iPart = 1:numPart
    startInd = (iPart-1)*numPartRotate;
    for r2 = 1:length(partRotationRange)
        allTemplateAffinityMatrix{r2, iPart} = templateAffinityMatrix{r2, iPart}(:)+startInd;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of auxiliary variables

%%% create filter bank
filters=[];
for iScale = 1:nScaleGabor
    f = MakeFilter(GaborScaleList(iScale),nOrient);
    for i=1:nOrient
        f_r{i} = single(real(f{i}));
        f_i{i} = single(imag(f{i}));
    end

    filters = [filters f_r f_i];
end

for iScale=1:nScaleDoG
    f0 = single(dog(DoGScaleList(iScale),0));
    filters = [filters f0];
end

numFilter = length(filters);
halfFilterSizes = zeros(size(filters));
for iF = 1:numFilter
    halfFilterSizes(iF)=(size(filters{iF},1)-1)/2;
end
overAllArea = sum((sx-2*halfFilterSizes).*(sy-2*halfFilterSizes));
Corr = CorrFilterFrame(filters); %calculate correlation among filters

%%%%%%%% optional: for drawing sketch in the learning process
[allFilterR, ~, filterSymbol] = MakeFilterBank(GaborScaleList, DoGScaleList, nOrient); % filter bank for function "drawSketch"
half = zeros(1, nOrient*nScaleGabor+nScaleDoG);
for i=1:(nOrient*nScaleGabor+nScaleDoG)
    half(i) = (size(allFilterR{1, i}, 1)-1)/2;
end
%%%%%%%%%%

files = dir(fullfile(inPath,'*.png'));
numImage=length(files);
disp('start filtering');

for iImg = 1:numImage
    copyfile(fullfile(inPath,files(iImg).name),fullfile(resultPath,'img',files(iImg).name));
    imageOriginal = imread(fullfile(inPath,files(iImg).name));
    img = imresize(imageOriginal,[sx, sy]);
    if ndims(img) == 3
        img = rgb2gray(img);
    end
    img = im2single(img);

    allSizex = zeros(1, numResolution);
    allSizey = zeros(1, numResolution);
    ImageMultiResolution = cell(1, numResolution);

    for resolution=1:numResolution
        resizeFactor = 1.0 + (resolution - originalResolution)*scaleStepSize;
        img2 = imresize(img, resizeFactor, 'nearest');  % images at multiple resolutions
        img2 = img2-mean(img2(:));
        img2 = img2/std(img2(:))*sqrt(sigsq);

        ImageMultiResolution{resolution} = img2;

        [sizex, sizey] = size(ImageMultiResolution{resolution});
        allSizex(resolution) = sizex;
        allSizey(resolution) = sizey;
    end

    % compute MAX1 map for images in different resolutions
    disp(['======> start filtering and maxing image ' num2str(iImg)]);
    tic
    [SUM1mapFind, MAX1mapFind] = applyfilterBank_MultiResolution_sparseV4(ImageMultiResolution, filters, halfFilterSizes, nOrient,...
        locationShiftLimit,orientShiftLimit,isLocalNormalize,isSeparate,localNormScaleFactor,thresholdFactor,nScaleGabor,nScaleDoG, sqrt(sigsq));  % if using local normalization, the last parameter is important.

    mapName = fullfile(cachePath,['SUMMAXmap-image' num2str(iImg) '.mat']);
    save(mapName, 'imageOriginal', 'ImageMultiResolution','SUM1mapFind', 'MAX1mapFind','allSizex', 'allSizey');

    disp(['filtering time: ' num2str(toc) ' seconds']);
end


%% Step 2: EM iteration
for it = 1:numEMIteration
    
    %%%% M-step
    template_name = sprintf([templatePath '/template_task%d_seed%d_iter%d.mat'], task_id, seed, it);
    load(template_name, 'clusters', 'MAX3scoreAll');
    
    savingFolder=fullfile(resultPath,['iteration' num2str(it) '/']);
    if ~exist(savingFolder)
        mkdir(savingFolder);
    end

    %%%% E-step
    disp(['E-step of iteration ' num2str(it)]);

    detectedCroppedSavingFolder=fullfile(savingFolder, 'detectedCropped/');
    if ~exist(detectedCroppedSavingFolder)
        mkdir(detectedCroppedSavingFolder);
    end

    morphedCroppedSavingFolder=fullfile(savingFolder, 'morphedCropped/');
    if ~exist(morphedCroppedSavingFolder)
        mkdir(morphedCroppedSavingFolder);
    end

    boundingBoxSavingFolder=fullfile(savingFolder, 'boundingBox/');
    if ~exist(boundingBoxSavingFolder)
        mkdir(boundingBoxSavingFolder);
    end

    MAX3scoreAll = zeros(numImage, numCluster);

    for iImg = 1:numImage
        mapName = fullfile(cachePath,['SUMMAXmap-image' num2str(iImg)]);
        imageLoaded = load(mapName);

        for c = 1:numCluster
            MAX3scoreAll(iImg, c) = detectObject(imageLoaded, clusters, c, iImg, numPart, numFilter, sx, sy, nOrient, part_sx, part_sy,...  % general parameters
                relativePartLocationRange, argmaxMethod, allTemplateAffinityMatrix, resolutionShiftLimit, ...   % parameters for local max
                rotationRange, partRotationRange, PartLocX, PartLocY, detectedCroppedSavingFolder, morphedCroppedSavingFolder, boundingBoxSavingFolder, RatioDisplacementSUM3);

        end
    end
    
    I_s = uint8(zeros([[100, 100], 3, numImage]));
    for iImg = 1:numImage
        I_s(:,:,:,iImg) = imread([boundingBoxSavingFolder '/' 'boundingBox-cluster-1-img-' sprintf('%04d', iImg) '.png']);
    end
    imSaveAsGif([boundingBoxSavingFolder '/boundingBox.gif'], I_s);

end

disp('done.');


