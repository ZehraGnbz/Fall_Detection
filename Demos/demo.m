% Clear command window, workspace, and close all figures
clc;
clear all;
close all;

% Step 1: Start the fall detection process
fallDetection();

function fallDetection()
    % Step 2: List of video files
    videoFiles = {'test3.mp4'};

    % Step 3: Process each video
    for i = 1:length(videoFiles)
        processVideo(videoFiles{i});
    end
end

function processVideo(videoFile)
    % Step 4: Create a video reader
    videoReader = VideoReader(videoFile);

    % Step 5: Load pre-trained YOLO v2 object detector
    detector = yolov2ObjectDetector('tiny-yolov2-coco');

    % Test the detector on a sample frame
    if hasFrame(videoReader)
        sampleFrame = readFrame(videoReader);
        [sampleBboxes, sampleScores, sampleLabels] = detect(detector, sampleFrame);
        detectedImg = insertObjectAnnotation(sampleFrame, 'rectangle', sampleBboxes, sampleLabels);
        figure; imshow(detectedImg); title('Sample detection');
    else
        disp('Error: Video file is empty or cannot be read.');
        return;
    end

    % Reopen the video file to start from the beginning
    videoReader = VideoReader(videoFile);

    % Read the first frame
    frame1 = readFrame(videoReader);

    % Step 7: Create optical flow object
    opticFlow = opticalFlowFarneback();

    while hasFrame(videoReader)
        % Step 8: Read the next frame
        frame2 = readFrame(videoReader);

        % Step 9: Detect humans in the frame
        [bboxes, scores, labels] = detect(detector, frame2, 'Threshold', 0.3); % Adjust the detection threshold

        % Step 10: Filter out non-human detections
        humanIdx = strcmp(labels, 'person');
        bboxes = bboxes(humanIdx, :);
        scores = scores(humanIdx);

        % Step 11: Display detections
        if isempty(bboxes)
            disp('No humans detected in this frame.');
        else
            disp(['Detected ', num2str(size(bboxes, 1)), ' humans in this frame.']);
        end

        % Visualize detections for debugging
        detectedFrame = insertObjectAnnotation(frame2, 'rectangle', bboxes, 'Person');
        figure; imshow(detectedFrame); title('Detections in current frame');

        % Step 12: Process each detected human
        for i = 1:size(bboxes, 1)
            % Step 12.1: Extract the region of interest (ROI) for the detected human
            bbox = bboxes(i, :);
            roiFrame1 = imcrop(frame1, bbox);
            roiFrame2 = imcrop(frame2, bbox);

            % Step 12.2: Compute optical flow within the ROI
            if ~isempty(roiFrame1) && ~isempty(roiFrame2)
                flow = computeOpticalFlow(opticFlow, roiFrame1, roiFrame2);

                % Step 12.3: Detect falls based on optical flow
                if detectFall(flow)
                    disp(['Fall detected in video: ', videoFile]);
                    % Display the frame with a visual indication
                    frame2 = insertShape(frame2, 'Rectangle', bbox, 'Color', 'red', 'LineWidth', 5);
                    imshow(frame2);
                    drawnow; % Update figure window
                    triggerAlarm();
                end
            end
        end

        % Step 13: Update frames
        frame1 = frame2;
    end
end

function flow = computeOpticalFlow(opticFlow, frame1, frame2)
    % Step 12.2.1: Convert frames to grayscale for optical flow computation
    gray1 = rgb2gray(frame1);
    gray2 = rgb2gray(frame2);

    % Step 12.2.2: Estimate optical flow using the Farneback method
    flow = estimateFlow(opticFlow, gray1);
    flow = estimateFlow(opticFlow, gray2);
end

function isFall = detectFall(flow)
    % Step 12.3.1: Compute the magnitude and angle of the flow vectors
    mag = sqrt(flow.Vx.^2 + flow.Vy.^2);
    angle = atan2(flow.Vy, flow.Vx);

    % Step 12.3.2: Threshold for detecting falls based on the vertical component
    verticalMotion = abs(sin(angle)) .* mag;
    fallThreshold = 10; % Adjust this threshold based on your dataset

    % Step 12.3.3: Check if the average vertical motion exceeds the threshold
    isFall = mean(verticalMotion(:)) > fallThreshold;
end

function triggerAlarm()
    % Step 12.3.4: Display an alarm message
    disp('Alarm! Fall detected.');
    % Optionally, add code to send an email or play a sound
end
