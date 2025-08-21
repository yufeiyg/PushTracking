import cv2
import numpy as np

depth_path = "/home/yufeiyang/Documents/BundleSDF/arm_data/T_shape_single/ob_0000001/depth_enhanced/00001.png"
# visualize this depth image
depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
depth_image = depth_image.astype(np.float32)  # Convert to meters
cv2.imshow("Depth Image", depth_image)
cv2.waitKey(0)
cv2.destroyAllWindows()