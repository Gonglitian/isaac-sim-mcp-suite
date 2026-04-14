#!/bin/bash
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
export HOME="${HOME:-/home/$(whoami)}"
export DISPLAY=:0
export PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

TOPIC="$1"
SAVE_PATH="$2"

/usr/bin/python3 << PYEOF
import rclpy, cv2, time, sys
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
rclpy.init()
node = rclpy.create_node('mcp_grab')
bridge = CvBridge()
done = [False]
def cb(msg):
    if not done[0]:
        img = bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imwrite('${SAVE_PATH}', img)
        done[0] = True
        print(f'GRABBED: ${SAVE_PATH} shape={img.shape}', file=sys.stderr)
qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
node.create_subscription(Image, '${TOPIC}', cb, qos)
t0 = time.time()
while rclpy.ok() and not done[0] and time.time()-t0 < 5:
    rclpy.spin_once(node, timeout_sec=0.5)
if not done[0]:
    print(f'TIMEOUT: no msg on ${TOPIC} after 5s', file=sys.stderr)
node.destroy_node()
rclpy.try_shutdown()
PYEOF
