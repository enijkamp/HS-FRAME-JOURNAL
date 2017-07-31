function para = config()

para.name = 'sequence';
para.dataPath = 'data-set/sequence/';
para.categoryName = 'apple';

para.method = 'two_stage';  % 'two_stage' : matching pursuit learning,  'one_stage': generative boosting learning

para.nPartCol = 2;
para.nPartRow = 2;
para.part_sx = 50;
para.part_sy = 50;

para.GaborScaleList = [0.2, 0.7];
para.DoGScaleList = [];

para.numCluster = 1;
para.numWavelet = 300;   % default 300, 370
para.numEMIteration = 10;
para.isLocalNormalize = true;

para.relativePartRotationRange=1*(-1:1);
para.relativePartLocationRange=1;

para.resolutionShiftLimit = 1;
para.rotateShiftLimit = 3;   % template rotation  from -rotateShiftLimit to rotateShiftLimit, eg. (1)-2:2 if rotateShiftLimit=2 (2)0 is without rotation
para.ratioDisplacementSUM3=0;   % default=0. Compute all values in SUM3 map

para.locationShiftLimit=3;  % arg-max
para.orientShiftLimit=1;    % arg-max

para.numResolution = 3;     

para.useSUM3 = 1;