import cv2
import numpy as np
from timeit import default_timer as timer

img = cv2.imread("Scotland-lowres.png", cv2.IMREAD_GRAYSCALE)  # queryiamge

n_kp = 30;


# Features
sift = cv2.SIFT_create(n_kp)
kp_image, desc_image = sift.detectAndCompute(img, None)


# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

img2 = cv2.imread("Scotland2-lowres.png", cv2.IMREAD_GRAYSCALE)


print("Starting keypoint search")
kp_image2, desc_img2 = sift.detectAndCompute(img2, None)
print("Starting keypoint match")
start=timer()
matches = flann.knnMatch(desc_image, desc_img2, k=2)
end=timer()
print(end-start)

print("Comparing %2d matches" % len(matches))

good_points = []

matches = [match for match in matches if len(match)==2]
matchesMask = [[0,0] for i in range(len(matches))]

for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        good_points.append(m)

print("Done comparing matches - retained %2d" % len(good_points))

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img,kp_image,img2,kp_image2,matches,None,**draw_params)
cv2.imshow("Matches",img3)

query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
train_pts = np.float32([kp_image2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
print("Homography found %2d %2d" % (len(query_pts),len(train_pts)))
matches_mask = mask.ravel().tolist()

# Perspective transform
h, w = img.shape
pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, matrix)

homography = cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3)

# Plot the keypoints used
kp_used = [kp_image2[m.trainIdx] for m in good_points];
img3=cv2.drawKeypoints(homography, kp_used, None)
cv2.imshow("Homography",img3)
cv2.waitKey()
