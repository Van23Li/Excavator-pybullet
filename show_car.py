import pybullet as p
import time
from time import sleep

def main():
    # 连接物理引擎
    p.connect(p.GUI)
    angle = p.addUserDebugParameter('Steering', -0.5, 0.5, 0)
    throttle = p.addUserDebugParameter('Throttle', -20, 20, 0)
    acceleration = p.addUserDebugParameter('Acceleration', -2, 2, 0)

    # Set camera
    # p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0, 0, 0])

    # Set gravity
    p.setGravity(0, 0, -10)

    wheel_indices = [1, 3, 4, 5]
    hinge_indices = [0, 2]
    car = p.loadURDF('descriptions/robot_descriptions/car_bullet/simplecar.urdf', [0, 0, 0.1])
    plane = p.loadURDF('descriptions/robot_descriptions/car_bullet/simpleplane.urdf')
    user_angle = 0
    user_throttle = 0
    while True:
        upKey = ord('w')
        downKey = ord('s')
        key = p.getKeyboardEvents()

        if upKey in key and key[upKey] & p.KEY_WAS_TRIGGERED:
            user_throttle += 1

        if downKey in key and key[downKey] & p.KEY_WAS_TRIGGERED:
            user_throttle -= 1

        leftKey = ord('a')
        rightKey = ord('d')
        if leftKey in key and key[leftKey] & p.KEY_WAS_TRIGGERED:
            user_angle -= 0.5
        elif rightKey in key and key[rightKey] & p.KEY_WAS_TRIGGERED:
            user_angle += 0.5

        accelKey = ord('q')
        deccelKey = ord('e')
        if deccelKey in key and key[deccelKey] & p.KEY_WAS_TRIGGERED:
            acceleration -= 0.5
        elif accelKey in key and key[accelKey] & p.KEY_WAS_TRIGGERED:
            acceleration += 0.5


        for joint_index in wheel_indices:
            p.setJointMotorControl2(car, joint_index,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=user_throttle)
        for joint_index in hinge_indices:
            p.setJointMotorControl2(car, joint_index,
                                    p.POSITION_CONTROL,
                                    targetPosition=user_angle)
        p.stepSimulation()

if __name__ == '__main__':
    main()
