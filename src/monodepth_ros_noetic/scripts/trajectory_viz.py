import rospy

from geometry_msgs.msg import Point, Quaternion
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from drone_msgs.msg import PelengatorGoal

class TrajectoryVizNode:
    """Нод-интерфейс для получения данных с пеленгатора и отправки в РОС
    """
    def __init__(self):
        self.request_rate = 2.0
        self.current_odom = None
        self.pose_arr = []
        
        self.request_timer = rospy.Timer(rospy.Duration(1 / self.request_rate), self.draw_trajectory)
        
        self.pub_marker = rospy.Publisher("/trajectory_viz/trajectory", Marker, queue_size=10)
        self.sub_odometry = rospy.Subscriber("/rtabmap/odom", Odometry, self.odom_cb)
        # self.calculation_timer = rospy.Timer(rospy.Duration(1.0/10.0), self.calculate)

    def odom_cb(self, odom):
        self.current_odom = odom
    
    def draw_trajectory(self, event):
        if self.current_odom is None:
            return
        self.pose_arr.append(self.odom_to_pose(self.current_odom))

        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()

        msg.color.a = 1.0
        msg.color.b = 0.0
        msg.color.g = 0.0
        msg.color.r = 1.0

        msg.id = 0
        msg.type = msg.LINE_STRIP
        msg.scale.x = 0.1

        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = 1.0

        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        for point in self.pose_arr:
            msg.points.append(point)
        
        self.pub_marker.publish(msg)

    @staticmethod
    def odom_to_pose(odom: Odometry) -> Point:
        pose = Point()

        pose.x = odom.pose.pose.position.x
        pose.y = odom.pose.pose.position.y
        pose.z = odom.pose.pose.position.z

        return pose



def main():
    # global peleng_pub

    rospy.init_node("trajectory_visualizer", anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    node = TrajectoryVizNode()

    while not rospy.is_shutdown():
        
        rate.sleep()
        rospy.spin()


if __name__ == '__main__':
    main()
