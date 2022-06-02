import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import sys


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)  --> search box size
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1) --> block size

    return: Disparity map, disp_map.shape = Left.shape
    """
    # Disparity map of CV2:
    # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=k_size)
    # disparity = stereo.compute(img_l, img_r)
    # plt.imshow(disparity, 'gray')
    # plt.show()

    img_l = np.pad(img_l, ((k_size // 2, k_size // 2), (k_size // 2, k_size // 2)), mode="edge")  # extend images to encapsulate the window of passage
    img_r = np.pad(img_r, ((k_size // 2, k_size // 2), (k_size // 2, k_size // 2)), mode="edge")
    disparity = np.zeros(img_l.shape)  # Init of returning array

    for x in range(k_size // 2, img_r.shape[0] - k_size // 2):
        for y in range(k_size // 2, img_r.shape[1] - k_size // 2):  # For every pixel in the original image (without padding)
            minMSE = sys.maxsize
            currDisparity = 0
            currWindow = img_l[x - k_size // 2:x + k_size // 2 + 1, y - k_size // 2:y + k_size // 2 + 1]  # Current window to compare
            for potentialDisparityInd in range(disp_range[0], disp_range[1]):
                if x + potentialDisparityInd < img_r.shape[0] - k_size // 2:
                    tempWindow = img_r[x + potentialDisparityInd - k_size // 2:x + potentialDisparityInd + k_size // 2 + 1, y - k_size // 2:y + k_size // 2 + 1]  # Window to compare to
                    new_array = (currWindow - tempWindow).astype(float)  # array of differences between the arrays
                    currMSE = np.sum(np.square(new_array))  # Sum of differences array squared is the error
                    if currMSE < minMSE:  # If the error is smaller than minMSE, update the minMSE and the corresponding disparity
                        minMSE = currMSE
                        currDisparity = potentialDisparityInd + disp_range[0]
            disparity[x][y] = currDisparity  # minimal error disparity
    return disparity


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    img_l = np.pad(img_l, ((k_size // 2, k_size // 2), (k_size // 2, k_size // 2)), mode="edge")  # extend images to encapsulate the window of passage
    img_r = np.pad(img_r, ((k_size // 2, k_size // 2), (k_size // 2, k_size // 2)), mode="edge")
    disparity = np.zeros(img_l.shape)

    for x in range(k_size // 2, img_r.shape[0] - k_size // 2):
        for y in range(k_size // 2, img_r.shape[1] - k_size // 2):  # For every pixel in the original image (without padding)
            maxNP = -sys.maxsize
            currDisparity = 0
            currWindow = img_l[x - k_size // 2:x + k_size // 2 + 1, y - k_size // 2:y + k_size // 2 + 1]  # Current window to compare
            for potentialDisparityInd in range(disp_range[0], disp_range[1]):
                if x + potentialDisparityInd < img_r.shape[0] - k_size // 2:
                    tempWindow = img_r[x + potentialDisparityInd - k_size // 2:x + potentialDisparityInd + k_size // 2 + 1, y - k_size // 2:y + k_size // 2 + 1]  # Window to compare to
                    currTempWinProduct = np.sum(currWindow * tempWindow)  # Product of the 2 matrices (numerator)
                    denominator = np.sqrt(np.sum(currWindow.astype(float) ** 2) * np.sum(tempWindow.astype(float) ** 2))  # Denominator of the NC equation
                    currNP = currTempWinProduct / denominator  # the current error
                    if currNP > maxNP:  # If error is smaller than the minimal error found up until now, update
                        maxNP = currNP
                        currDisparity = potentialDisparityInd + disp_range[0]
            disparity[x][y] = currDisparity  # minimal error disparity
    return disparity


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destination image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """

    # Algorithm adapted from the answer at the following link:
    # https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog?newreg=740dd4c5f2514e04a6d57016be0c2759

    mat = np.zeros((src_pnt.shape[0] * 2, 9))
    for i in range(src_pnt.shape[0]):
        x, y = src_pnt[i][0], src_pnt[i][1]  # The points used (for code readability)
        xP, yP = dst_pnt[i][0], dst_pnt[i][1]
        # Constructing the matrix for the homography calculations
        mat[i * 2][0] = -x
        mat[i * 2][1] = -y
        mat[i * 2][2] = -1
        mat[i * 2][6] = xP * x
        mat[i * 2][7] = xP * y
        mat[i * 2][8] = xP

        mat[i * 2 + 1][3] = -x
        mat[i * 2 + 1][4] = -y
        mat[i * 2 + 1][5] = -1
        mat[i * 2 + 1][6] = yP * x
        mat[i * 2 + 1][7] = yP * y
        mat[i * 2 + 1][8] = yP

    v = np.linalg.svd(mat)[2]  # using numpy's svd function to return the svd tuple. only the final value of the svd is needed (the last vector in V)
    homographyMat = v[-1].reshape((3, 3))  # reshaping the last vector to match the 3x3 matrix
    homographyMat /= homographyMat[2][2]  # the bottom right corner of the homography matrix must be 1: normalizing

    # Calculating error
    ones = np.ones((src_pnt.shape[0], 1))
    src_pnt = np.append(src_pnt, ones, axis=1)
    dst_pnt = np.append(dst_pnt, ones, axis=1)

    error = 0
    for i in range(src_pnt.shape[0]):
        error += np.sqrt(np.sum(homographyMat.dot(src_pnt[i]) / homographyMat.dot(src_pnt[i])[-1] - dst_pnt[i]) ** 2)
    return homographyMat, error


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """

    """
    How to use: After running the function, 2 images (one after the other) will pop up. On each,
    select 4+ points to calculate and press q to pass to the next image or exit. The function
    will then calculate the results
    """

    clicked = []

    def waitForKeyPress():
        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("q"):
                break
        # close all open windows
        cv2.destroyAllWindows()

    def onclick(event, x, y, flags, param):
        # if the left mouse button was clicked, append the (x, y) coordinates into the list
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append([x, y])

    # Recording the src points
    cv2.namedWindow("src_img")
    cv2.setMouseCallback("src_img", onclick)
    cv2.imshow("src_img", src_img)
    waitForKeyPress()
    srcPoints = clicked
    clicked = []

    # Recording the dest points
    cv2.namedWindow("dst_img")
    cv2.setMouseCallback("dst_img", onclick)
    cv2.imshow("dst_img", dst_img)
    waitForKeyPress()
    destPoints = clicked

    ################################
    # Points have been recorded
    ################################

    srcPoints = np.array([[200, 100], [200, 1000], [1700, 100], [1700, 1000]])
    destPoints = np.array([[276, 303], [276, 800], [1666, 154], [1666, 703]])

    minX = np.min(srcPoints[:, 0])  # Identifying the corners of the rectangle to fit
    maxX = np.max(srcPoints[:, 0])
    minY = np.min(srcPoints[:, 1])
    maxY = np.max(srcPoints[:, 1])

    homographyMat = cv2.findHomography(np.array(srcPoints), np.array(destPoints))[0]  # Calculating homography matrix using cv2 function
    warpedImg = dst_img.copy()
    for x in range(minX, maxX):  # For every pixel in the defined range
        for y in range(minY, maxY):
            newPoint = np.matmul(homographyMat, np.array([[x], [y], [1]]))  # multiplying the homography matrix by the point
            newPoint /= newPoint[2][0]  # Normalizing the point (last value should be 1)
            warpedImg[int(np.round(newPoint[1][0]))][int(np.round(newPoint[0][0]))] = src_img[y][x]  # Inserting the point into the new image

    plt.imshow(warpedImg)
    plt.show()
