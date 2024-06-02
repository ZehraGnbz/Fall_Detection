% Close all figures, clear all variables, and clear the command window
close all;
clear all;
clc;

% Complete MATLAB script for person and fall detection in two videos sequentially

% Function to process a single video file
function processVideo(videoFile, outputFile)
    % Load the video file
    videoReader = VideoReader(videoFile);
    
    % Create a video writer to save annotated video
    outputVideo = VideoWriter(outputFile, 'Motion JPEG AVI');
    open(outputVideo);
    
    % Create people detector using ACF
    peopleDetector = peopleDetectorACF();
    
    % Create optical flow object
    opticFlow = opticalFlowLK('NoiseThreshold', 0.01);
    
    % Initialize tracking and fall detection
    peopleTracker = multiObjectTracker('FilterInitializationFcn', @initKalmanFilter, 'AssignmentThreshold', 30);
    fallDetectionThreshold = 30; % Threshold for fall detection based on position change
    previousPositions = [];
    previousFlow = [];
    firstFrame = true; % Flag to ensure tracker is updated first

    while hasFrame(videoReader)
        frame = readFrame(videoReader);
        
        % Detect people
        bboxes = detect(peopleDetector, frame);
        
        % Update tracker
        if ~isempty(bboxes)
            if firstFrame
                % Initialize tracks on the first frame
                initializeTracks(peopleTracker, bboxes);
                firstFrame = false;
            else
                % Predict new locations of tracks
                tracks = predictTracksToTime(peopleTracker, videoReader.CurrentTime);
                
                % Detection-to-track assignment
                assignments = assignDetectionsToTracks(tracks, bboxes, 0.2);
                
                % Update track with assigned detections
                for i = 1:size(assignments, 1)
                    trackID = assignments(i, 1);
                    detectionIdx = assignments(i, 2);
                    updateTrack(peopleTracker, trackID, bboxes(detectionIdx, :));
                end
            end
        end
        
        % Optical flow
        grayFrame = rgb2gray(frame);
        flow = estimateFlow(opticFlow, grayFrame);
        
        % Display annotated frame
        annotatedFrame = insertObjectAnnotation(frame, 'rectangle', bboxes, 'Person');
        
        % Detect falls
        for i = 1:numel(tracks)
            if isActive(tracks(i))
                % Get current and previous position
                currentPos = tracks(i).Measurement;
                if i <= size(previousPositions, 1)
                    previousPos = previousPositions(i, :);
                    if norm(currentPos - previousPos) > fallDetectionThreshold
                        % Additional checks with optical flow
                        if isFallDetected(flow, previousFlow)
                            % Annotate frame with fall detection
                            annotatedFrame = insertText(annotatedFrame, tracks(i).State(1:2), 'Fall Detected', 'BoxColor', 'red', 'FontSize', 18);
                            disp(['Fall detected at time: ', num2str(videoReader.CurrentTime), ' seconds']);
                        end
                    end
                end
                previousPositions(i, :) = currentPos;
            end
        end
        
        % Update previous flow
        previousFlow = flow;
        
        % Write the frame to the output video
        writeVideo(outputVideo, annotatedFrame);
    end
    
    close(outputVideo);
    disp(['Processing complete for video: ', videoFile, '. Check the output video for fall detections.']);
end

% Process the first video
processVideo('test6.mp4', 'output_test6_with_falls.avi');

% Process the second video
processVideo('test7.mp4', 'output_test7_with_falls.avi');

% Helper Functions

function isFall = isFallDetected(flow, previousFlow)
    % Function to detect fall based on optical flow
    isFall = false;
    % Implement fall detection logic here based on optical flow
    if ~isempty(previousFlow)
        flowMagnitude = sum((flow.Magnitude - previousFlow.Magnitude).^2);
        if flowMagnitude > 50 % Example threshold for detecting a fall
            isFall = true;
        end
    end
end

function kalmanFilter = initKalmanFilter()
    % Initialize a Kalman filter for tracking
    kalmanFilter = vision.KalmanFilter('MotionModel', 'ConstantVelocity', ...
        'State', [0, 0, 0, 0], ...
        'MeasurementNoise', 1e-5, ...
        'ProcessNoise', [1, 1, 1, 1] * 1e-5, ...
        'MeasurementModel', [1 0 0 0; 0 0 1 0]);
end

function initializeTracks(tracker, bboxes)
    % Initialize tracks with detected bounding boxes
    for i = 1:size(bboxes, 1)
        bbox = bboxes(i, :);
        % Initialize a new track
        correct(tracker, bbox);
    end
end

function updateTrack(tracker, trackID, bbox)
    % Update the track with a new detection
    correct(tracker, trackID, bbox);
end
