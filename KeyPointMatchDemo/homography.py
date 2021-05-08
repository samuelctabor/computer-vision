import cv2
import numpy as np
from timeit import default_timer as timer

img = cv2.imread("Scotland-lowres.png", cv2.IMREAD_GRAYSCALE)  # queryiamge

#cv2.imshow("Homography", img)
#cv2.waitKey()


#cap = cv2.VideoCapture(0)
n_kp = 30;


# Features
#sift = cv2.xfeatures2d.SIFT_create(n_kp)
sift = cv2.SIFT_create(n_kp)
kp_image, desc_image = sift.detectAndCompute(img, None)


# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

#_, frame = cap.read()
#grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage

grayframe = cv2.imread("Scotland2-lowres.png", cv2.IMREAD_GRAYSCALE)
frame = grayframe

print("Starting keypoint search")
kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
print("Starting keypoint match")
start=timer()
matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
end=timer()
print(end-start)

print("Comparing %2d matches" % len(matches))

good_points = []

for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_points.append(m)

print("Done comparing matches - retained %2d" % len(good_points))

query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
print("Homography found %2d %2d" % (len(query_pts),len(train_pts)))
matches_mask = mask.ravel().tolist()
# Perspective transform
h, w = img.shape
pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, matrix)

homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

# Plot the keypoints used
kp_used = [kp_grayframe[m.trainIdx] for m in good_points];
img3=cv2.drawKeypoints(homography, kp_used, None)
cv2.imshow("Homography",img3)
cv2.waitKey()

