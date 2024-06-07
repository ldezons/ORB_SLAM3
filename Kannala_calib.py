import rosbag
from cv_bridge import CvBridge
import cv2
import numpy as np

def extract_images_from_rosbag(bag_file, image_topic):
    bag = rosbag.Bag(bag_file, 'r')
    bridge = CvBridge()
    images = []

    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        images.append(cv_image)

    bag.close()
    return images

def find_calibration_points(images, pattern_size=(9, 6)):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints

def calibrate_camera(images, pattern_size=(9, 6), image_size=(640, 480)):
    objpoints, imgpoints = find_calibration_points(images, pattern_size)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = []
    tvecs = []
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        image_size,
        K,
        D,
        rvecs,
        tvecs,
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW,
        (cv2.TermCriteria_EPS + cv2.TermCriteria_COUNT, 100, 1e-6)
    )
    return K, D

def main():
    bag_file = 'rosbag/calib.bag'
    image_topic = '/camera_bridge/cam_down/image_mono'
    pattern_size = (9, 6)
    image_size = (640, 400)  # Set this to the size of your images

    images = extract_images_from_rosbag(bag_file, image_topic)
    K, D = calibrate_camera(images, pattern_size, image_size)
    
    print("Camera matrix (K):\n", K)
    print("Distortion coefficients (D):\n", D)

if __name__ == '__main__':
    main()
