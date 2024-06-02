classdef MultiObjectTrackerOpticalFlow < handle
    properties
        OpticalFlow;
        Points = [];
        PointIds = [];
        NextId = 1;
    end
    
    methods
        function this = MultiObjectTrackerOpticalFlow()
            this.OpticalFlow = opticalFlowLK('NoiseThreshold', 0.009);
        end
        
        function initializePoints(this, I)
            % Detect feature points in the image
            points = detectMinEigenFeatures(I);
            this.Points = points.Location;
            
            % Assign unique IDs to each point
            this.PointIds = (1:size(this.Points, 1))';
        end
        
        function track(this, I)
            % Estimate optical flow
            flow = this.OpticalFlow.estimateFlow(I);
            
            % Update the points based on the optical flow
            for i = 1:size(this.Points, 1)
                this.Points(i, :) = this.Points(i, :) + [flow.Vx(round(this.Points(i, 2)), round(this.Points(i, 1))), ...
                                                         flow.Vy(round(this.Points(i, 2)), round(this.Points(i, 1)))];
            end
            
            % Remove points that move out of frame
            validIdx = this.Points(:, 1) > 0 & this.Points(:, 1) <= size(I, 2) & ...
                       this.Points(:, 2) > 0 & this.Points(:, 2) <= size(I, 1);
            this.Points = this.Points(validIdx, :);
            this.PointIds = this.PointIds(validIdx);
            
            % Reinitialize points if too few are left
            if size(this.Points, 1) < 10
                this.initializePoints(I);
            end
        end
        
        function tracks = getTracks(this)
            % Return the tracked points
            tracks = struct('points', this.Points, 'ids', this.PointIds);
        end
    end
end
