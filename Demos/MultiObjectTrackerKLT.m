classdef MultiObjectTrackerKLT < handle
    properties
        PointTracker; 
        Bboxes = [];
        BoxIds = [];
        BoxLabels = {}; % Initialize as a cell array
        Points = [];
        PointIds = [];
        NextId = 1;
        BoxScores = [];
    end
    
    methods
        function this = MultiObjectTrackerKLT()
            this.PointTracker = vision.PointTracker('MaxBidirectionalError', 2);
        end
        
        function addDetections(this, I, bboxes, labels)
            % Convert to grayscale if the image is RGB
            if size(I, 3) == 3
                I = rgb2gray(I);
            end
            
            for i = 1:size(bboxes, 1)
                box = bboxes(i, :);
                % Debugging: print each bounding box
                disp('Processing bounding box:');
                disp(box);
                
                if numel(box) ~= 4
                    error('Bounding box must be a 4-element array [x, y, width, height].');
                end
                
                boxIdx = this.findMatchingBox(box);
                
                if isempty(boxIdx)
                    this.Bboxes = [this.Bboxes; box];
                    points = detectMinEigenFeatures(I, 'ROI', box);
                    points = points.Location;
                    this.BoxIds(end+1) = this.NextId;
                    idx = ones(size(points, 1), 1) * this.NextId;
                    this.PointIds = [this.PointIds; idx];
                    if (~exist('labels', 'var') || isempty(labels{i}))
                        labels{i} = num2str(this.NextId);
                    end
                    this.BoxLabels{end+1} = labels{i}; % Store labels as cell array elements
                    this.NextId = this.NextId + 1;
                    this.Points = [this.Points; points];
                    this.BoxScores(end+1) = 1;
                    
                else
                    currentBoxScore = this.deleteBox(boxIdx);
                    this.Bboxes = [this.Bboxes; box];
                    points = detectMinEigenFeatures(I, 'ROI', box);
                    points = points.Location;
                    this.BoxIds(end+1) = boxIdx;
                    idx = ones(size(points, 1), 1) * boxIdx;
                    this.PointIds = [this.PointIds; idx];
                    if (~exist('labels', 'var') || isempty(labels{i}))
                        labels{i} = num2str(boxIdx);
                    end
                    this.BoxLabels{end+1} = labels{i}; % Store labels as cell array elements
                    this.Points = [this.Points; points];                    
                    this.BoxScores(end+1) = currentBoxScore + 1;
                end
            end
            
            minBoxScore = -2;
            this.BoxScores(this.BoxScores < 3) = this.BoxScores(this.BoxScores < 3) - 0.5;
            boxesToRemoveIds = this.BoxIds(this.BoxScores < minBoxScore);
            while ~isempty(boxesToRemoveIds)
                this.deleteBox(boxesToRemoveIds(1));
                boxesToRemoveIds = this.BoxIds(this.BoxScores < minBoxScore);
            end
            
            if this.PointTracker.isLocked()
                this.PointTracker.setPoints(this.Points);
            else
                if ~isempty(this.Points)
                    this.PointTracker.initialize(this.Points, I);
                end
            end
        end
        
        function track(this, I)
            % Convert to grayscale if the image is RGB
            if size(I, 3) == 3
                I = rgb2gray(I);
            end

            if this.PointTracker.isLocked()
                [newPoints, isFound] = this.PointTracker.step(I);
                this.Points = newPoints(isFound, :);
                this.PointIds = this.PointIds(isFound);
                this.generateNewBoxes();
                if ~isempty(this.Points)
                    this.PointTracker.setPoints(this.Points);
                end
            end
        end
        
        function tracks = getTracks(this)
            numBoxes = size(this.Bboxes, 1);
            tracks = struct('bbox', cell(1, numBoxes), 'id', cell(1, numBoxes), 'label', cell(1, numBoxes));
            for i = 1:numBoxes
                tracks(i).bbox = this.Bboxes(i, :);
                tracks(i).id = this.BoxIds(i);
                tracks(i).label = this.BoxLabels{i}; % Access labels as cell array elements
            end
        end
    end
    
    methods(Access=private)        
        function boxIdx = findMatchingBox(this, box)
            boxIdx = [];
            for i = 1:size(this.Bboxes, 1)
                area = rectint(this.Bboxes(i,:), box);                
                if area > 0.2 * this.Bboxes(i, 3) * this.Bboxes(i, 4)
                    boxIdx = this.BoxIds(i);
                    return;
                end
            end           
        end
        
        function currentScore = deleteBox(this, boxIdx)
            this.Bboxes(this.BoxIds == boxIdx, :) = [];
            this.BoxLabels(this.BoxIds == boxIdx) = []; % Use correct indexing for cell array
            this.Points(this.PointIds == boxIdx, :) = [];
            this.PointIds(this.PointIds == boxIdx) = [];
            currentScore = this.BoxScores(this.BoxIds == boxIdx);
            this.BoxScores(this.BoxIds == boxIdx) = [];
            this.BoxIds(this.BoxIds == boxIdx) = [];
        end
        
        function generateNewBoxes(this)
            oldBoxIds = this.BoxIds;
            oldLabels = this.BoxLabels;
            oldScores = this.BoxScores;
            this.BoxIds = unique(this.PointIds);
            this.BoxLabels = cell(size(this.BoxIds)); % Initialize as cell array
            numBoxes = numel(this.BoxIds);
            this.Bboxes = zeros(numBoxes, 4);
            this.BoxScores = zeros(numBoxes, 1);
            for i = 1:numBoxes
                points = this.Points(this.PointIds == this.BoxIds(i), :);
                newBox = getBoundingBox(points);
                this.Bboxes(i, :) = newBox;
                this.BoxScores(i) = oldScores(oldBoxIds == this.BoxIds(i));
                this.BoxLabels{i} = oldLabels{oldBoxIds == this.BoxIds(i)};
            end
        end 
    end
end

function bbox = getBoundingBox(points)
x1 = min(points(:, 1));
y1 = min(points(:, 2));
x2 = max(points(:, 1));
y2 = max(points(:, 2));
bbox = [x1 y1 x2 - x1 y2 - y1];
end
