import numpy as np
import matplotlib.pyplot as plt
import cv2
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

    img_l = np.pad(img_l, ((k_size // 2, k_size // 2), (k_size // 2, k_size // 2)), mode="edge")
    img_r = np.pad(img_r, ((k_size // 2, k_size // 2), (k_size // 2, k_size // 2)), mode="edge")
    disparity = np.zeros(img_l.shape)

    for x in range(k_size // 2, img_r.shape[0] - k_size // 2):
        for y in range(k_size // 2, img_r.shape[1] - k_size // 2):
            minMSE = sys.maxsize
            currDisparity = 0
            currWindow = img_l[x - k_size // 2:x + k_size // 2 + 1, y - k_size // 2:y + k_size // 2 + 1]
            for potentialDisparityInd in range(disp_range[0], disp_range[1]):
                if x + potentialDisparityInd < img_r.shape[0] - k_size // 2:
                    tempWindow = img_r[x + potentialDisparityInd - k_size // 2:x + potentialDisparityInd + k_size // 2 + 1, y - k_size // 2:y + k_size // 2 + 1]
                    new_array = (currWindow - tempWindow).astype(float)
                    currMSE = np.sum(np.square(new_array))
                    if currMSE < minMSE:
                        minMSE = currMSE
                        currDisparity = potentialDisparityInd + disp_range[0]
            disparity[x][y] = currDisparity
    return disparity


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    img_l = np.pad(img_l, ((k_size // 2, k_size // 2), (k_size // 2, k_size // 2)), mode="edge")
    img_r = np.pad(img_r, ((k_size // 2, k_size // 2), (k_size // 2, k_size // 2)), mode="edge")
    disparity = np.zeros(img_l.shape)

    for x in range(k_size // 2, img_r.shape[0] - k_size // 2):
        for y in range(k_size // 2, img_r.shape[1] - k_size // 2):
            maxNP = -sys.maxsize
            currDisparity = 0
            currWindow = img_l[x - k_size // 2:x + k_size // 2 + 1, y - k_size // 2:y + k_size // 2 + 1]
            for potentialDisparityInd in range(disp_range[0], disp_range[1]):
                if x + potentialDisparityInd < img_r.shape[0] - k_size // 2:
                    tempWindow = img_r[x + potentialDisparityInd - k_size // 2:x + potentialDisparityInd + k_size // 2 + 1, y - k_size // 2:y + k_size // 2 + 1]
                    currTempWinProduct = np.sum(currWindow * tempWindow)
                    denominator = np.sqrt(np.sum(currWindow.astype(float) ** 2) * np.sum(tempWindow.astype(float) ** 2))
                    currNP = currTempWinProduct / denominator
                    if currNP > maxNP:
                        maxNP = currNP
                        currDisparity = potentialDisparityInd + disp_range[0]
            disparity[x][y] = currDisparity
    return disparity


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """

    # Algorithm adapted from the answer at the following link:
    # https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog?newreg=740dd4c5f2514e04a6d57016be0c2759

    mat = np.zeros((src_pnt.shape[0] * 2, 9))
    for i in range(src_pnt.shape[0]):
        x, y = src_pnt[i][0], src_pnt[i][1]
        xP, yP = dst_pnt[i][0], dst_pnt[i][1]
        mat[i * 2][0] = x
        mat[i * 2][1] = y
        mat[i * 2][2] = 1
        mat[i * 2][6] = -xP * x
        mat[i * 2][7] = -xP * y
        mat[i * 2][8] = -xP

        mat[i * 2 + 1][3] = x
        mat[i * 2 + 1][4] = y
        mat[i * 2 + 1][5] = 1
        mat[i * 2 + 1][6] = -yP * x
        mat[i * 2 + 1][7] = -yP * y
        mat[i * 2 + 1][8] = -yP
    u, s, v = np.linalg.svd(mat)
    homographyMat = v[-1].reshape((3, 3))
    homographyMat /= homographyMat[2][2]

    # Calculating error
    hInv = np.linalg.inv(homographyMat)
    error = 0
    for i in range(src_pnt.shape[0]):
        x = np.array([src_pnt[i][0], src_pnt[i][1], 1]).T
        xP = np.array([dst_pnt[i][0], dst_pnt[i][1], 1]).T
        error += (x - np.matmul(hInv, xP)).astype(float)**2 + (xP - np.matmul(homographyMat, x)).astype(float)**2

    return homographyMat, np.sum(error)


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######

    # out = dst_img * mask + src_out * (1 - mask)
    # plt.imshow(out)
    # plt.show()
