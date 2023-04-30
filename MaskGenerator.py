# Face Mask generator

import cv2
import numpy as np


class MaskGenerator:
    def __init__(self):
        self.target = {}

    # Find convex hull from the landmarks found in the image
    def find_convex_hull(self, points):
        hull = []
        hullIndex = cv2.convexHull(np.array(points), clockwise=False, returnPoints=False)

        addPoints = [
            [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
            [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
            [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
            [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
            [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
        ]

        hullIndex = np.concatenate((hullIndex, addPoints))
        for i in range(0, len(hullIndex)):
            hull.append(points[int(hullIndex[i][0])])

        return hull, hullIndex

    # Check if a point is inside a rectangle
    def rectContains(self, rect, point):
        return point[0] >= rect[0] and point[1] >= rect[1] and point[0] <= rect[2] and point[1] <= rect[3]

    # Calculate Delaunay triangles for set of points
    # Returns the vector of indices of 3 points for each triangle
    def calculateDelaunayTriangles(self, rect, points):
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert((int(p[0]), int(p[1])))

        # Get Delaunay triangulation
        triangleList = subdiv.getTriangleList()

        delaunay = []

        for t in triangleList:
            # Get triangle as a list of 3 points
            pt = [(t[i], t[i+1]) for i in [0,2,4]]

            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            if self.rectContains(rect, pt1) and self.rectContains(rect, pt2) and self.rectContains(rect, pt3):
                ind = []
                for j in range(0, 3):
                    for k in range(0, len(points)):
                        if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                            ind.append(k)

                if len(ind) == 3:
                    delaunay.append((ind[0], ind[1], ind[2]))

        return delaunay

    # Apply affine transform calculated using srcTri and dstTri to src
    def applyAffineTransform(self, src, srcTri, dstTri, size):

        warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
        dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return dst

    # Warps triangular regions from img1 and img2 to img
    def warpTriangle(self, img1, img2, t1, t2):
        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        # Offset points by left top corner of the respective rectangles
        t1Rect = []
        t2Rect = []
        t2RectInt = []

        for i in range(0, 3):
            t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        size = (r2[2], r2[3])
        img2Rect = self.applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
        img2Rect = img2Rect * mask

        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect

    def calculateTargetInfo(self, target_image, target_alpha, target_landmarks):

        hull, hullIndex = self.find_convex_hull(target_landmarks)

        sizeImg1 = target_image.shape
        rect = (0, 0, sizeImg1[1], sizeImg1[0])
        dt = self.calculateDelaunayTriangles(rect, hull)

        self.target["image"] = target_image
        self.target["width"] = sizeImg1[1]
        self.target["height"] = sizeImg1[0]
        self.target["alpha"] = target_alpha
        self.target["landmarks"] = target_landmarks
        self.target["hull"] = hull
        self.target["hullIndex"] = hullIndex
        self.target["dt"] = dt

    def applyTargetMask(self, actual_img, actual_landmarks):
        warped_img = np.copy(actual_img)

        hull2 = []
        for i in range(0, len(self.target["hullIndex"])):
            hull2.append(actual_landmarks[self.target["hullIndex"][i][0]])

        mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
        mask1 = cv2.merge((mask1, mask1, mask1))
        img1_alpha_mask = cv2.merge((self.target["alpha"], self.target["alpha"], self.target["alpha"]))

        # Warp the triangles
        for i in range(0, len(self.target["dt"])):
            t1 = []
            t2 = []
            for j in range(0, 3):
                t1.append(self.target["hull"][self.target["dt"][i][j]])
                t2.append(hull2[self.target["dt"][i][j]])

            self.warpTriangle(self.target["image"], warped_img, t1, t2)
            self.warpTriangle(img1_alpha_mask, mask1, t1, t2)

        mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
        mask2 = (255.0, 255.0, 255.0) - mask1

        # Alpha blending of the two images
        temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
        temp2 = np.multiply(actual_img, (mask2 * (1.0 / 255)))

        #cv2.imshow('temp1', np.uint8(temp1))
        #cv2.imshow('temp2', np.uint8(temp2))

        output = temp1 + temp2

        self.temp1 = temp1
        self.mask1 = mask1
        self.temp2 = temp2

        return np.uint8(output)

    def applyTargetMaskToTarget(self, actual_landmarks):
        t_w, t_h = (self.target["width"], self.target["height"])
        # 0. Calculate homography actual_landmarks -> target_landmarks
        pts_src = np.array([[p[0], p[1]] for p in actual_landmarks])
        dst_src = np.array([[p[0], p[1]] for p in self.target["landmarks"]])
        h, _ = cv2.findHomography(pts_src, dst_src)

        # 1. Apply homography to actual_img
        im_out_temp1 = cv2.warpPerspective(self.temp1, h, (t_w, t_h))
        im_out_mask1 = cv2.warpPerspective(self.mask1, h, (t_w, t_h))

        # 2. Overlap result in target_image
        mask2 = (255.0, 255.0, 255.0) - im_out_mask1

        # 3. Apply homography in the opposite direction
        target_image = np.copy(self.target["image"])

        '''
        pts_src = pts_src.reshape(-1,1,2).astype(np.float32)
        tmp = cv2.perspectiveTransform(pts_src, h)
        pts_src = dst_src
        dst_src = tmp.astype(np.int32)

        h, _ = cv2.findHomography(pts_src, dst_src)
        target_image = cv2.warpPerspective(target_image, h, (t_w, t_h))
        '''

        # 4. Alpha blending of the two images
        temp2 = np.multiply(target_image, (mask2 * (1.0 / 255)))
        output = im_out_temp1 + temp2

        return np.uint8(output)
