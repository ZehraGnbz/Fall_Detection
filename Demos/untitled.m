% Clear command window, workspace, and close all figures
clc;
clear all;
close all;

% Start the fall and slip detection process
fallAndSlipDetection();

function fallAndSlipDetection()
    % List of video files
    videoFiles = {'test7.mp4'}; 

    % Process each video
    for i = 1:length(videoFiles)
        processVideo(videoFiles{i});
    end
end

function processVideo(videoFile)
    % Create a video reader
    videoReader = VideoReader(videoFile);

    % Create optical flow object
    opticFlow = opticalFlowFarneback();

    % Create a video player for displaying frames
    videoPlayer = vision.VideoPlayer('Position', [100, 100, 680, 520]);

    % Read the first frame
    if hasFrame(videoReader)
        frame1 = readFrame(videoReader);
    else
        disp('Error: Video file is empty or cannot be read.');
        return;
    end

    % Initialize tracking variables
    isPersonDetected = false;
    bbox = [];

    % Initialize time tracking variables
    fallTime = [];
    slipTime = [];
    fallFrame = [];
    slipFrame = [];

    while hasFrame(videoReader)
        % Read the next frame
        frame2 = readFrame(videoReader);
        
        % Convert to grayscale
        grayFrame1 = rgb2gray(frame1);
        grayFrame2 = rgb2gray(frame2);

        % Compute optical flow
        flow = estimateFlow(opticFlow, grayFrame2);

        % Detect falls and slips
        [isFall, isSlip] = detectFallOrSlip(flow, bbox);

        % Detect and track the person using YOLO if not already detected
        if ~isPersonDetected
            yolov2Detector = yolov2ObjectDetector('tiny-yolov2-coco');
            [bboxes, scores, labels] = detect(yolov2Detector, frame2);
            humanIdx = strcmp(labels, 'person');
            bboxes = bboxes(humanIdx, :);
            scores = scores(humanIdx);

            % Filter detections based on a score threshold
            scoreThreshold = 0.5;
            highScoreIdx = scores > scoreThreshold;
            bboxes = bboxes(highScoreIdx, :);
            scores = scores(highScoreIdx);

            if ~isempty(bboxes)
                bbox = bboxes(1, :);
                isPersonDetected = true;
            end
        else
            % Track the person using the previous bounding box
            [bbox, isPersonDetected] = trackPerson(frame1, frame2, bbox);
        end

        % Visualize the detection result
        if isFall
            disp(['Fall detected in video: ', videoFile, ' at time: ', num2str(videoReader.CurrentTime), ' seconds']);
            fallTime = [fallTime; videoReader.CurrentTime];
            fallFrame = frame2;
            frame2 = insertText(frame2, [10, 10], 'Fall Detected', 'FontSize', 18, 'BoxColor', 'red', 'BoxOpacity', 0.6);
        elseif isSlip
            disp(['Slip detected in video: ', videoFile, ' at time: ', num2str(videoReader.CurrentTime), ' seconds']);
            slipTime = [slipTime; videoReader.CurrentTime];
            slipFrame = frame2;
            frame2 = insertText(frame2, [10, 10], 'Slip Detected', 'FontSize', 18, 'BoxColor', 'blue', 'BoxOpacity', 0.6);
        end

        % Draw bounding box if person is detected
        if isPersonDetected
            frame2 = insertShape(frame2, 'Rectangle', bbox, 'LineWidth', 3, 'Color', 'green');
            frame2 = insertText(frame2, [bbox(1), bbox(2)-10], 'Person', 'FontSize', 12, 'BoxColor', 'green', 'BoxOpacity', 0.6);
        end

        % Display the current frame
        step(videoPlayer, frame2);

        % Update frames
        frame1 = frame2;
    end

    % Release the video player
    release(videoPlayer);

    % Display detected fall and slip times
    if ~isempty(fallTime)
        disp('Fall times (in seconds):');
        disp(fallTime);
        figure;
        imshow(fallFrame);
        title('Fall Detected Frame');
    end
    if ~isempty(slipTime)
        disp('Slip times (in seconds):');
        disp(slipTime);
        figure;
        imshow(slipFrame);
        title('Slip Detected Frame');
    end
end

function [isFall, isSlip] = detectFallOrSlip(flow, bbox)
    % Compute the magnitude and angle of the flow vectors
    mag = sqrt(flow.Vx.^2 + flow.Vy.^2);
    angle = atan2(flow.Vy, flow.Vx);

    % Threshold for detecting falls based on the vertical component
    verticalMotion = abs(sin(angle)) .* mag;
    fallThreshold = 1.5; % Adjust this threshold based on your dataset
    isFall = mean(verticalMotion(:)) > fallThreshold;

    % Threshold for detecting slips based on the horizontal component
    horizontalMotion = abs(cos(angle)) .* mag;
    slipThreshold = 1.5; % Adjust this threshold based on your dataset
    isSlip = mean(horizontalMotion(:)) > slipThreshold;

    % Additional fall detection based on person's orientation
    if ~isempty(bbox)
        % Check if the person is more horizontal than usual
        aspectRatio = bbox(3) / bbox(4); % Width / Height
        if aspectRatio > 1.5 % Adjust this threshold based on your dataset
            isFall = true;
        end
    end
end

function [bbox, isPersonDetected] = trackPerson(frame1, frame2, bbox)
    % Define the point tracker
    persistent pointTracker;
    if isempty(pointTracker)
        pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
        initialize(pointTracker, bbox2points(bbox), frame1);
    end
    
    % Track the points
    [points, isFound] = step(pointTracker, frame2);
    visiblePoints = points(isFound, :);

    % Update the bounding box based on the tracked points
    if size(visiblePoints, 1) >= 2
        newBBox = points2bbox(visiblePoints);
        bbox = newBBox;
        isPersonDetected = true;
    else
        isPersonDetected = false;
    end
end

function bbox = points2bbox(points)
    % Convert points to bounding box
    xCoords = points(:, 1);
    yCoords = points(:, 2);
    bbox = [min(xCoords), min(yCoords), max(xCoords) - min(xCoords), max(yCoords) - min(yCoords)];
end

function points = bbox2points(bbox)
    % Convert bounding box to points
    x1 = bbox(1);
    y1 = bbox(2);
    x2 = x1 + bbox(3);
    y2 = y1 + bbox(4);
    points = [x1 y1; x2 y1; x2 y2; x1 y2];
end
