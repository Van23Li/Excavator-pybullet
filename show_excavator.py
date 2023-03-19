import numpy as np
import pybullet as p
import time
import pybullet_data

# 连接物理引擎
physicsCilent = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# Set camera
# p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0, 0, 0])

# Set simulation-step
hz = 1000
delta_t = 1.0 / hz
p.setTimeStep(delta_t)
p.setRealTimeSimulation(0)

# Set gravity
p.setGravity(0, 0, -9.8)

# 添加资源路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())
urdf_path = "descriptions/robot_descriptions/excavator_bullet/excavatorplus.urdf"

# URDF: base_link; 1huizhuantai; 2dongbi; 3dougan; 4chandou
controlled_joints = [1, 2, 3]
numJoints = len(controlled_joints)

# Load the plane
planeId = p.loadURDF("plane.urdf", [0, 0, -6.0])
p.changeDynamics(planeId, -1, restitution=0.9)

startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=True) # If you wanna fix the walking device,
                                          # set "useFixedBase" to True, otherwise, set it to False

# reset the joint
ul = np.array([1.0360, -0.7987, 0.5633])
ll = np.array([-0.7987, -2.9552, -2.7163])
q0 = ll
p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in q0])

# 开始一千次迭代，也就是一千次交互，每次交互后停顿1/240
for i in range(1000):
    p.stepSimulation()
    time.sleep(1 / 240)

# 获取位置与方向四元数
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print("-" * 20)
print(f"机器人的位置坐标为:{cubePos}\n机器人的朝向四元数为:{cubeOrn}")
print("-" * 20)

# 断开连接
p.disconnect()