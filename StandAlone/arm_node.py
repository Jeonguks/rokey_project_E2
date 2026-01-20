import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class ArmTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__("arm_traj_pub")
        self.pub = self.create_publisher(
            JointTrajectory, "/arm_controller/joint_trajectory", 10
        )
        self.timer = self.create_timer(2.0, self.send_traj)

        self.joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

    def send_traj(self):
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        p = JointTrajectoryPoint()
        p.positions = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0]
        p.time_from_start.sec = 2

        traj.points.append(p)
        self.pub.publish(traj)


def main():
    rclpy.init()
    node = ArmTrajectoryPublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
