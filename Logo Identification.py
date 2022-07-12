import cv2
import numpy as np

img1 = cv2.imread('logo.png')   #logo to be identified
img2 = cv2.imread('images.png')  #raw image
grayimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create(nfeatures = 5000)

#getting features
kp_img1, desc_img1 = sift.detectAndCompute(img1, None)
img1_features = cv2.drawKeypoints(img1, kp_img1, img1)
cv2.imshow("img1 features", img1_features)

kp_grayimg2, desc_grayimg2 = sift.detectAndCompute(grayimg2, None)
img2_features = cv2.drawKeypoints(grayimg2, kp_grayimg2, grayimg2)
cv2.imshow("img2 features", img2_features)

#Feature Matching
index_params = dict(algorithm = 0, trees = 5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc_img1, desc_grayimg2, k=2)

#For taking only good matches
good_pts=[]
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good_pts.append(m)

#Drawing Matches Made
img3 = cv2.drawMatches(img1, kp_img1, grayimg2, kp_grayimg2, good_pts, img1,2)
cv2.imshow("Matches",img3)
cv2.waitKey()

#Logo is found only if there are sufficient matches else matches made are not accurate and logo is not found. 
if len(good_pts) > 10:
    #homography
    query_pts = np.float32([kp_img1[m.queryIdx].pt for m in good_pts]).reshape(-1,1,2)
    train_pts = np.float32([kp_grayimg2[m.trainIdx].pt for m in good_pts]).reshape(-1,1,2)
    matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    #perspective transform
    h = img1.shape[0]
    w = img1.shape[1]
    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, matrix)

    #Drawing a box to higlight found logo
    img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result",img2)
else:
    print("Logo Not Found")

cv2.waitKey()

