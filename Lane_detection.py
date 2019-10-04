import numpy as np
import math
import cv2 as cv
from cannyEdge import CannyEdgeDetect


def do_segment(frame):
    # Since an image is a multi-directional array containing the relative intensities of each pixel
    # in the image, we can use frame.shape to return a tuple:
    # [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has.
    # Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
    height, width = frame.shape

    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    #polygons = np.array([ [(0, height), (900, height), (380, 290)] ])
    polygons = np.array([
        [(0, height), (width-10, height),(340, 260),  (360,260)]
    ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv.bitwise_and(frame, mask)
    return segment

def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    x = 0
    lines = lines[1:]
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)
        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients
        # which describe the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < -0.5:
            left.append((slope, y_intercept))
        elif slope > 0.5:
        #else:
            right.append((slope, y_intercept))
        # if x == 0: print("slope: ", slope, "b: ", y_intercept)
        x+=1
    # Averages out all the values for left and right into a single slope
    # and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    print("left: \n", left_line)
    print("right: \n", right_line)
    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize


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
    # keep track of exactly which points contributed a vote to each Hough bin:
    x_min_points = np.full(accumulator.shape, np.argmax(x_idxs))
    y_min_points = np.zeros(accumulator.shape)
    x_max_points = np.zeros(accumulator.shape)
    y_max_points = np.full(accumulator.shape, np.argmax(y_idxs))
    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            # rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1
            if y > y_min_points[rho, t_idx]:
                y_min_points[rho, t_idx] = y
                x_min_points[rho, t_idx] = x
            elif y < y_max_points[rho, t_idx]:
                y_max_points[rho, t_idx] = y
                x_max_points[rho, t_idx] = x

    value_threshold = 35
    max_votes = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    # building al list of the params
    lines = np.empty((1,4),dtype=int)
    x = 0 #------- need to be erase -------
    for i in range (max_votes.shape[0]):
        for j in range (max_votes.shape[1]):
            if accumulator[i,j] > value_threshold:
                max_votes[i,j] = accumulator[i,j]
                # transform this max_votes + theta + rhos matrices to the output.
                # output is: 2d array, number of rows = number of lines, each matrix-line contains 2 point of the line.
                r = rhos[i]
                theta = thetas[j]
                """if x == 0:
                    print("r(distance) = ", rhos[i])
                    print("theta(angle) = ", theta)
                    print("np.cos(theta) = ", np.cos(theta))
                    print("np.sin(theta) = ",np.sin(theta))"""
                # a = np.cos(theta)
                # b = np.sin(theta)
                # x0 = a * r
                # y0 = b * r
                # x1 = int(x0 + 1000 * (-b))
                # y1 = int(y0 + 1000 * (a))
                # x2 = int(x0 - 1000 * (-b))
                # y2 = int(y0 - 1000 * (a))
                # https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/
                x1 = x_min_points[i,j]
                y1 = y_min_points[i,j]
                x2 = x_max_points[i,j]
                y2 = y_max_points[i,j]
                points = np.array([x1, y1, x2, y2], dtype=int)
                # print("points: \n" , points)
                lines = np.vstack((lines, points))
                if x == 0:
                    print(lines[0,:])
                    print("i = ", i, " j = ", j)
                    print("dist = ", r , "angle + ", theta)
                x+=1#--------------to erase-----------------

    # usualy implementation of hough transform return max_votes, thetas, rhos
    return lines



"""frame = cv.imread("shot1.jpg")
canny = do_canny(frame)
print("canny done")
# cv.imshow("canny", canny)
# cv.waitKey()
# plt.imshow(frame)
# plt.show()
segment = do_segment(canny)
print("segment done")
cv.imshow("segment", segment)
cv.waitKey()
#hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
# hough = hough_line(segment)
lines = hough_line(segment)
print("hough done")
print(lines)
# print("hough.shape", lines.shape)
# Averages multiple detected lines from hough into one line for left border of lane and one line for right border of lane
lines = calculate_lines(frame, lines)
# Visualizes the lines
lines_visualize = visualize_lines(frame, lines)
cv.imshow("hough", lines_visualize)
# Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
output = cv.addWeighted(frame, 0.9, lines_visualize, 1, 1)
# Opens a new window and displays the output frame
cv.imshow("output", output)
cv.waitKey()"""

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("testVideo.mp4")
while (cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # canny = do_canny(frame)
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    do_canny = CannyEdgeDetect()
    canny = do_canny.find_edges(grayFrame)
    segment = do_segment(canny)
    # hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
    hough = hough_line(segment)
    # Averages multiple detected lines from hough into one line for left border of lane and one line for right border of lane
    # lines = calculate_lines(frame, hough)
    lines = hough
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
cv.destroyAllWindows()
