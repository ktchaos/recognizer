import sys
import argparse
import cv2
from matplotlib import pyplot as plt
from Libraries.ExtractKeypoints.ExtractKeypoints import extractKeypoints

# 1 -> "/Users/catarinaserrano/Desktop/UFPB/SignatureRecognition/Data/signature.jpg"
# 2 -> "/Users/catarinaserrano/Desktop/UFPB/SignatureRecognition/Data/ref-signature.jpg"

def main():
    print('----|| INIT MODULE ||----')
    img1 = cv2.imread('Data/signature.jpg', cv2.IMREAD_GRAYSCALE)
    kp1, des1 = extractKeypoints(img1)
    img2 = cv2.imread('Data/ref-signature.jpg', cv2.IMREAD_GRAYSCALE)
    kp2, des2 = extractKeypoints(img2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda match: match.distance)

    img4 = cv2.drawKeypoints(img1, kp1, outImage=None)
    img5 = cv2.drawKeypoints(img2, kp2, outImage=None)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img4)
    axarr[1].imshow(img5)
    plt.show()

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)
    plt.imshow(img3)
    plt.show()

    score = 0
    for match in matches:
        score += match.distance
    if score / len(matches) < 20:
        print("RESULT: Signature match with score = {}".format(100-(score / len(matches))))
    else:
        print("RESULT: Signature does not match.")
    print('----|| MODULE ENDED ||----')


main()
