%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%                        Zehra Betül Günbaz                                %
%                      Senior Project 2023-2024                            %
%                       Fall/Slip Detection                                %
%                Thanks to Aykut Yıldız Assistant Professor                %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function fall_detection_with_pose_estimation()
    % Close all figures, clear workspace, and clear command window
    close all;
    clear all;
    clc;

    % Disable all warnings
    warningState = warning('off', 'all');
    
    % Enable required add-ons
    matlab.addons.enableAddon('f3e11ae6-430a-45cc-a939-18cc7d4c14fb'); % Human Pose Estimation with Deep Learning
    matlab.addons.enableAddon('4c7a0681-0457-48ba-85a2-55076024b3b5'); % OpenPose - wrapper - MATLAB

    % Ask the user to select video files
    [fileNames, pathName] = uigetfile({'*.mp4'}, 'Select Video Files', 'MultiSelect', 'on');

    if isequal(fileNames, 0)
        disp('User cancelled the operation.');
        % Restore previous warning state
        warning(warningState);
        return;
    end

    if ischar(fileNames)
        videoFiles = {fullfile(pathName, fileNames)};
    else
        videoFiles = fullfile(pathName, fileNames);
    end

    allFrames = [];
    allOriginalFrames = [];
    frameRates = [];
    videoDurations = [];

    for k = 1:length(videoFiles)
        videoReader = VideoReader(videoFiles{k});
        frames = [];
        originalFrames = [];
        frameRate = videoReader.FrameRate;
        videoDuration = videoReader.Duration;
        
        frameRates = [frameRates; frameRate];
        videoDurations = [videoDurations; videoDuration];
        
        while hasFrame(videoReader)
            frame = readFrame(videoReader);
            originalFrames = cat(4, originalFrames, frame);
            % Apply preprocessing: noise cancellation and Gaussian smoothing
            preprocessedFrame = preprocessFrame(frame);
            % Resize to 1:1 aspect ratio (256x256) as required by the pose estimation model
            resizedFrame = imresize(preprocessedFrame, [256 256]);
            frames = cat(4, frames, resizedFrame);
        end
        
        allFrames = cat(4, allFrames, frames);
        allOriginalFrames = cat(4, allOriginalFrames, originalFrames);
    end

    % Initialize variables
    frameCount = size(allFrames, 4);
    poseKeyPoints = cell(frameCount, 1);  % Use cell array to store keypoints for multiple people
    disappearFrames = zeros(frameCount, 1);  % To track disappearance of keypoints

    % Initialize video player
    videoPlayer = vision.VideoPlayer('Position', [100, 100, 680, 520]);

    % Initialize people detector
    peopleDetector = vision.PeopleDetector;
    
    % Initialize side profile face detector
    sideProfileDetector = vision.CascadeObjectDetector('ProfileFace');

    % Initialize optical flow
    opticFlow = opticalFlowFarneback;

    % Load the pre-trained pose estimation network using the PoseEstimator class
    poseEstimator = PoseEstimator('MATFile', 'simplePoseNet.mat', 'NetworkName', 'simplePoseNet', 'SkeletonConnectionMap', 'SkeletonConnectionMap');

    possibleFalls = [];

    % Initialize best frame variables
    highestScore = 0;
    bestFrame = [];
    bestOriginalFrame = [];
    bestFrameTime = 0;
    prevKeyPoints = [];
    prevVelocity = zeros(17, 1); % Initialize for 17 keypoints
    keyPoints = []; % Initialize keyPoints

    ignoreFrames = 45; % Ignore the first 45 frames for best result calculations
    fallDetectedFrames = [];

    % Variables for tracking the best fall frame
    bestFallScore = 0;
    bestFallFrame = [];
    bestFallBboxes = [];
    bestTimestamp = '';

    % Define the range for the last 30% of frames
    last30PercentStart = floor(0.7 * frameCount) + 1;

    % Initialize video writer to save the processed video
    [~, videoFileName, ~] = fileparts(videoFiles{1});
    outputVideoFileName = fullfile('/Users/macbook/Desktop/Falling footage/Footage test/Results', [videoFileName, '_processed.mp4']);
    videoWriter = VideoWriter(outputVideoFileName, 'MPEG-4');
    videoWriter.FrameRate = frameRate;
    open(videoWriter);

    for i = 1:frameCount
        % Determine which video's frame rate to use
        frameRate = frameRates(floor((i - 1) / (frameCount / length(videoFiles))) + 1);

        % Perform human detection
        frame = allOriginalFrames(:,:,:,i);
        bboxes = peopleDetector(frame);
        
        % Perform side profile detection if no person is detected
        if isempty(bboxes)
            try
                sideProfileBBoxes = sideProfileDetector(frame);
                if ~isempty(sideProfileBBoxes)
                    bboxes = sideProfileBBoxes;
                    frame = insertShape(frame, 'Rectangle', bboxes, 'LineWidth', 3, 'Color', 'yellow');
                end
            catch
                disp(['Side profile detection error at frame ', num2str(i)]);
            end
        end
        
        if ~isempty(bboxes)
            % Draw bounding box around detected person
            frame = insertShape(frame, 'Rectangle', bboxes, 'LineWidth', 3, 'Color', 'yellow');
            
            % Normalize bounding boxes and resize images to fit the network input
            [croppedImages, croppedBBoxes] = poseEstimator.normalizeBBoxes(frame, bboxes);
            
            % Perform human pose estimation within the bounding boxes
            keyPoints = poseEstimator.detectPose(croppedImages);
            keyPoints = reshape(keyPoints, [], 3, size(keyPoints, 4));  % Reshape keypoints to 17x3xN
            
            % Store keypoints for each detected person
            poseKeyPoints{i} = keyPoints;
            
            % Draw keypoints on the frame
            if size(croppedBBoxes, 1) == size(keyPoints, 3)  % Ensure there's a matching number of bounding boxes and keypoints
                frame = poseEstimator.visualizeKeyPointsMultiple(frame, keyPoints, croppedBBoxes);
            end
            
            % Check for sudden shrinkage or rotation change in the bounding box
            if i > 1
                prevBBox = bboxes;
                [newBboxes, scores] = peopleDetector(frame);
                if ~isempty(newBboxes)
                    shrinkage = (newBboxes(3) * newBboxes(4)) / (prevBBox(3) * prevBBox(4));
                    rotationChange = abs(prevBBox(1) - newBboxes(1));
                    
                    % Define criteria for fall detection based on shrinkage or rotation change
                    if shrinkage < 0.5 || rotationChange > 50
                        if i > ignoreFrames  % Ignore falls detected in the first 45 frames
                            frame = insertText(frame, [10, 10], 'Fall/Slip Detected', 'FontSize', 18, 'BoxColor', 'red', 'BoxOpacity', 0.6);
                            possibleFalls = [possibleFalls; i];
                        end
                    end
                end
            end
        else
            % Track disappearance of keypoints
            disappearFrames(i) = disappearFrames(i) + 1;
            if disappearFrames(i) > 10
                if i > ignoreFrames  % Ignore falls detected in the first 45 frames
                    frame = insertText(frame, [10, 10], 'Fall/Slip Detected (Disappeared)', 'FontSize', 18, 'BoxColor', 'red', 'BoxOpacity', 0.6);
                    possibleFalls = [possibleFalls; i];
                end
            end
        end
        
        % Perform motion detection using optical flow
        if i > 1
            previousFrame = allFrames(:,:,:,i-1);
            flow = estimateFlow(opticFlow, rgb2gray(previousFrame));
            flowMagnitude = sqrt(flow.Vx.^2 + flow.Vy.^2);
            motionDetected = mean(flowMagnitude(:)) > 1; % Threshold for motion detection
            
            if motionDetected
                frame = insertText(frame, [10, 40], 'Motion Detected', 'FontSize', 18, 'BoxColor', 'green', 'BoxOpacity', 0.6);
            end
        else
            flowMagnitude = zeros(size(frame, 1), size(frame, 2)); % Initialize for the first frame
        end
        
        % Detect skeleton corruption
        if i > 1 && ~isempty(poseKeyPoints{i-1}) && ~isempty(poseKeyPoints{i})
            for j = 1:size(poseKeyPoints{i-1}, 3)
                if size(poseKeyPoints{i-1}, 1) >= j && size(poseKeyPoints{i}, 1) >= j && ...
                        sum(poseKeyPoints{i-1}(:,3,j)) > 0 && sum(poseKeyPoints{i}(:,3,j)) > 0 % Check for valid keypoints
                    corruptedSkeleton = detectSkeletonCorruption(poseKeyPoints{i-1}(:,:,j), poseKeyPoints{i}(:,:,j));
                    if corruptedSkeleton
                        if i > ignoreFrames  % Ignore falls detected in the first 45 frames
                            frame = insertText(frame, [10, 70], 'Skeleton Corruption Detected', 'FontSize', 18, 'BoxColor', 'blue', 'BoxOpacity', 0.6);
                            possibleFalls = [possibleFalls; i];
                        end
                    end
                end
            end
        end
        
        % Check for falls and find the best frame
        if ~isempty(keyPoints)
            for j = 1:size(keyPoints, 3)
                if size(keyPoints, 1) >= j && size(prevKeyPoints, 1) >= j
                    % Ensure keypoint arrays are of the same size
                    currentKeyPoints = keyPoints(:, :, j);
                    if size(currentKeyPoints, 1) == size(prevKeyPoints, 1)
                        isFall = checkFall(currentKeyPoints, prevKeyPoints, prevVelocity, frameRate);

                        if isFall && (isempty(fallDetectedFrames) || (i - fallDetectedFrames(end) > 45))
                            fallScore = sum(currentKeyPoints(:, 3)); % Sum of confidence scores as fall score
                            fallDetectedFrames = [fallDetectedFrames; i];

                            % Update best frame if conditions are met
                            if fallScore > highestScore && i > ignoreFrames && i >= last30PercentStart  % Exclude first 45 frames and focus on last 30% of frames
                                highestScore = fallScore;
                                bestFrame = allFrames(:,:,:,i);
                                bestOriginalFrame = allOriginalFrames(:,:,:,i);
                                bestFrameTime = i / frameRate;
                            end

                            if max(currentKeyPoints(:, 3)) > bestFallScore && i > ignoreFrames && i >= last30PercentStart  % Exclude first 45 frames and focus on last 30% of frames
                                bestFallScore = max(currentKeyPoints(:, 3));
                                bestFallFrame = frame;
                                if j <= size(bboxes, 1)
                                    bestFallBboxes = bboxes(j, :);
                                end
                                bestTimestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');
                            end

                            disp(['Possible Fall detected at frame ', num2str(i)]);
                        end
                    end
                end
            end
        end
        
        prevKeyPoints = keyPoints;
        if ~isempty(keyPoints)
            currentKeyPoints = keyPoints(:, :, 1);  % Assuming only one person is detected
            if size(currentKeyPoints, 1) == size(prevKeyPoints, 1)
                prevVelocity = sqrt((currentKeyPoints(:, 1) - prevKeyPoints(:, 1)).^2 + (currentKeyPoints(:, 2) - prevKeyPoints(:, 2)).^2) * frameRate;
            end
        else
            prevVelocity = zeros(17, 1); % Reset to zeros if keyPoints is empty
        end
        
        % Write the frame to the video file
        writeVideo(videoWriter, frame);
        
        % Display the frame in the video player
        videoPlayer(frame);
        pause(1/frameRate); % Adjust the pause to match the video frame rate
    end

    % Close video writer
    close(videoWriter);

    % Release video player
    release(videoPlayer);

    % Display the best fall frame in a new window and save images
    if ~isempty(bestFallFrame)
        if ~isempty(bestFallBboxes)
            bestFallFrame = insertShape(bestFallFrame, 'Rectangle', bestFallBboxes, 'Color', 'red');
        end
        bestFallFrame = insertText(bestFallFrame, [10, 10], ['Fall Detected: ' bestTimestamp], 'FontSize', 18, 'TextColor', 'red');
        outputImageFile = fullfile('/Users/macbook/Desktop/Falling footage/Footage test/Results', [videoFileName, '_fall_detected.jpg']);
        imwrite(bestFallFrame, outputImageFile);

        % Display the final detected frame
        figure;
        imshow(bestFallFrame);
        title('Most Accurate Fall Detected Frame');
        
        if ~isempty(possibleFalls)
            disp(['Most likely there is a fall at frame number ', num2str(possibleFalls(end))]);
            likelyFallFrame = allOriginalFrames(:,:,:,possibleFalls(end));
            likelyFallFrame = insertText(likelyFallFrame, [10, 10], 'Most Likely Fall Frame', 'FontSize', 18, 'TextColor', 'red');
            imwrite(likelyFallFrame, fullfile('/Users/macbook/Desktop/Falling footage/Footage test/Results', [videoFileName, '_processed2.jpg']));
            figure;
            imshow(likelyFallFrame);
            title('Frame Where Fall is Most Likely');
        end
    else
        disp('No fall detected in the video.');
    end

    disp('Processing completed.');

    % Restore previous warning state
    warning(warningState);
end

function corrupted = detectSkeletonCorruption(prevKeypoints, currentKeypoints)
    % Detect skeleton corruption by checking for missing or significantly changed keypoints
    threshold = 50; % Example threshold for significant change
    corrupted = false;
    
    for i = 1:min(size(prevKeypoints, 1), size(currentKeypoints, 1))
        if prevKeypoints(i, 3) > 0 && currentKeypoints(i, 3) == 0
            corrupted = true;
            return;
        end
        
        if currentKeypoints(i, 3) > 0
            distance = sqrt((prevKeypoints(i, 1) - currentKeypoints(i, 1))^2 + (prevKeypoints(i, 2) - currentKeypoints(i, 2))^2);
            if distance > threshold
                corrupted = true;
                return;
            end
        end
    end
end

function isFall = checkFall(currentKeypoints, prevKeypoints, prevVelocity, frameRate)
    isFall = false;
    
    if isempty(prevKeypoints) || isempty(prevVelocity)
        return;
    end
    
    % Check if the person is horizontal
    shoulderDistance = abs(currentKeypoints(2, 2) - currentKeypoints(5, 2));
    verticalThreshold = 30; % Example threshold for vertical distance (increased for robustness)
    
    if shoulderDistance < verticalThreshold
        isFall = true;
        return;
    end
    
    % Check for significant change in keypoints
    threshold = 100; % Example threshold for significant change (increased for robustness)
    for i = 1:min(size(currentKeypoints, 1), size(prevKeypoints, 1))
        if currentKeypoints(i, 3) > 0 && prevKeypoints(i, 3) > 0
            distance = sqrt((prevKeypoints(i, 1) - currentKeypoints(i, 1))^2 + (prevKeypoints(i, 2) - currentKeypoints(i, 2))^2);
            if distance > threshold
                isFall = true;
                return;
            end
        end
    end

    % Check for sudden acceleration/deceleration
    currentVelocity = sqrt((currentKeypoints(:, 1) - prevKeypoints(:, 1)).^2 + (currentKeypoints(:, 2) - prevKeypoints(:, 2)).^2) * frameRate;
    acceleration = abs(currentVelocity - prevVelocity);
    accelerationThreshold = 200; % Example threshold for acceleration (increased for robustness)
    if any(acceleration > accelerationThreshold)
        isFall = true;
        return;
    end
end

function preprocessedFrame = preprocessFrame(frame)
    % Convert the frame to grayscale
    grayFrame = rgb2gray(frame);
    % Apply Gaussian smoothing
    smoothedFrame = imgaussfilt(grayFrame, 2);
    % Apply noise cancellation (e.g., median filtering)
    preprocessedFrame = medfilt2(smoothedFrame, [3 3]);
    % Convert back to RGB
    preprocessedFrame = repmat(preprocessedFrame, [1 1 3]);
end
