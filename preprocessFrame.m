
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%                        Zehra Betül Günbaz                                %
%                      Senior Project 2023-2024                            %
%                       Fall/Slip Detection                                %
%                Thanks to Aykut Yıldız Assistant Professor                %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
