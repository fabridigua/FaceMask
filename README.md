# FaceMask - A Real-time Face Morphing Tool

---

This project aims to use some computer vision concepts to build a tool for face morphing with a real-time webcam video stream.

![](.\result\facemask_cage.gif)

![facemask_obama](.\result\facemask_obama.gif)

![facemask_putin](.\result\facemask_putin.gif)

![facemask_trump](.\result\facemask_trump.gif)



### Requirements

- Python 3.8
- OpenCV 4.7.0
- Mediapipe 0.9.3

### Projects Steps

![img](https://cdn-images-1.medium.com/v2/resize:fit:1000/1*216UrE2MikYgVibEAjZUtA.png)

### How it works

The application consists into 6 steps:

1. **Target face landmarks** detection
2. **Source** (webcam) **face landmarks** detection
3. **Delaunay triangulation** calculation
4. **Warping of triangles** of the two faces
5. **Alpha blending** of the warped image
6. **Homography** calculation and **perspective warping**



For more details see the related medium article.