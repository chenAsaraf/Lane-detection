import numpy as np
import math
import cv2 as cv

"""need to implement by ourselves:
 1. cv.cvtColor
 2. cv.GaussianBlur
 3. cv.Canny(Canny edge detector)
"""
def do_canny(frame):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv.Canny(blur, 50, 150)
    return canny



def do_segment(frame):
    # Since an image is a multi-directional array containing the relative intensities of each pixel
    # in the image, we can use frame.shape to return a tuple:
    # [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has.
    # Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
    height = frame.shape[0]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([
                            [(0, height), (800, height), (380, 290)]
                        ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv.bitwise_and(frame, mask)
    return segment
    
  
  def hough_line(segment):
    # Rho and Theta ranges
    angle_step = 1
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = segment.shape
    # diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, num=diag_len * 2.0, dtype=int) # Return evenly spaced numbers over a specified interval.
    # --- maby erase dtype int from above-----

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(segment)  # (row, col) indexes to edges
    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            # rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    value_threshold = 40
    
    max_votes = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    # building al list of the output
    lines = np.empty((1,4),dtype=int)
    for i in range (max_votes.shape[0]):
        for j in range (max_votes.shape[1]):
            if accumulator[i,j] > value_threshold:
                max_votes[i,j] = accumulator[i,j]
                # transform this max_votes + theta + rhos matrices to the output.
                # output is: 2d array, number of rows = number of lines, each matrix-line contains 2 point of the line.
                r = rhos[i]
                theta = thetas[j]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * r
                y0 = b * r
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                # https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/
                points = np.array([x1, y1, x2, y2])
                lines = np.vstack((lines, points))
                
    # usualy implementation of hough transform return max_votes, thetas, rhos
    return lines
    
    
# ----------------- Testing the project with one shot ----------------- #

frame = cv.imread("shot1.jpg")
canny = do_canny(frame)
print("canny done")
cv.imshow("canny", canny)
cv.waitKey()
segment = do_segment(canny)
print("segment done")
cv.imshow("segment", segment)
cv.waitKey()
# hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
lines = hough_line(segment)
# Visualizes the lines
lines_visualize = visualize_lines(frame, lines)
cv.imshow("hough", lines_visualize)
# Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
output = cv.addWeighted(frame, 0.9, lines_visualize, 1, 1)
# Opens a new window and displays the output frame
cv.imshow("output", output)
cv.waitKey()
    
# ----------------- Testing the project with video ----------------- #

# The video feed is read in as a VideoCapture object
"""cap = cv.VideoCapture("testVideo.mp4")
while (cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    canny = do_canny(frame)
    segment = do_segment(canny)
    lines = hough_lines(segment)
    # Averages multiple detected lines from hough into one line for left border of lane and one line for right border of lane
    # lines = calculate_lines(frame, hough)
    # Visualizes the lines
    lines_visualize = visualize_lines(frame, lines)
    cv.imshow("hough", lines_visualize)
    # Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
    output = cv.addWeighted(frame, 0.9, lines_visualize, 1, 1)
    # Opens a new window and displays the output frame
    cv.imshow("output", output)
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()"""
