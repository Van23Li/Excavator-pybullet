import numpy as np
import pybullet as p
import time
import pybullet_data
from ruckig import InputParameter, Ruckig, Trajectory, Result

def main():
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
    controlled_joints = [0, 1, 2, 3]
    numJoints = len(controlled_joints)

    # Load the plane
    planeId = p.loadURDF("plane.urdf", [0, 0, -6.0])
    p.changeDynamics(planeId, -1, restitution=0.9)

    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=True) # If you wanna fix the walking device,
                                            # set "useFixedBase" to True, otherwise, set it to False

    # reset the joint
    ul = np.array([-2.5, 1.0360, -0.7987, 0.5633])
    ll = np.array([2.5, -0.7987, -2.9552, -2.7163])
    q0 = 0.5 * (ul + ll)
    q0_dot = np.zeros(len(q0))
    qd = ul
    qd_dot = np.zeros(len(qd))
    # p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in qd])

    # Plan the trajectory using (https://github.com/pantor/ruckig)
    trajectory = get_traj_from_ruckig(q0=q0, q0_dot=q0_dot, qd=qd, qd_dot=qd_dot, Dof = numJoints)

    # Extract the trajectory planned by ruckig
    t0, tf = 0, trajectory.duration
    plan_time = tf - t0
    sample_t = np.arange(0, tf, delta_t)
    n_steps = sample_t.shape[0]
    traj_data = np.zeros([3, n_steps, 4])
    for i in range(n_steps):
        for j in range(3):
            traj_data[j, i] = trajectory.at_time(sample_t[i])[j]

    # reset the joint
    q0 = traj_data[0, 0]
    p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in q0])
    tt = 0

    # Simulation
    while(True):
        ref = trajectory.at_time(tt)
        p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]], targetVelocities=[[q0_i] for q0_i in ref[1]])
        p.stepSimulation()
        tt = tt + delta_t
        time.sleep(delta_t)
        if tt > trajectory.duration + 2.0:
            break

    # 断开连接
    p.disconnect()

def get_traj_from_ruckig(q0, q0_dot, qd, qd_dot, Dof):
    inp = InputParameter(Dof)

    inp.current_position = q0
    inp.current_velocity = q0_dot
    inp.current_acceleration = np.zeros(Dof)

    inp.target_position = qd
    inp.target_velocity = qd_dot
    inp.target_acceleration = np.zeros(Dof)

    # Set limits
    inp.max_velocity = np.array([2.1750, 2.1750, 2.1750, 2.1750])
    inp.max_acceleration = np.array([15, 7.5, 10, 12.5]) * 0.9
    inp.max_jerk = np.array([10000, 10000, 10000, 10000])*0.01

    otg = Ruckig(Dof)
    trajectory = Trajectory(Dof)
    _ = otg.calculate(inp, trajectory)
    return trajectory

if __name__ == '__main__':
    main()