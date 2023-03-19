import numpy as np
import time
import math
import pybullet as p
import pybullet_data
from pathlib import Path
from sys import path
from ruckig import InputParameter, Ruckig, Trajectory, Result
# Path to the build directory including a file simila r to 'ruckig.cpython-37m-x86_64-linux-gnu'.
build_path = Path(__file__).parent.absolute().parent / 'build'
path.insert(0, str(build_path))

def main():
    # Define problem parameters
    ul = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]) - 0.25
    ll = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) + 0.25

    # initial joint position
    q0 = 0.5*(ul+ll)
    q0_dot = np.zeros(7)
    g = -9.81

    # data path
    robot_path = "robot_data/fixed_base_panda_robotiq_5_joint_dense_1_dataset_15"
    urdf_path = "descriptions/robot_descriptions/franka_panda_bullet/panda.urdf"
    # pybullet client to compute robot fk
    clid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    robot = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
    query_position_dthetaz = [0.85, 0, 0]      #  [0.85, 0.6*math.pi,0]
    query_position_xyz = np.array([query_position_dthetaz[0]*np.cos(query_position_dthetaz[1]),
                                   query_position_dthetaz[0]*np.sin(query_position_dthetaz[1]),
                                   query_position_dthetaz[2]])
    q_candidates,phi_candidates,throw_candidates, qdot_candidates = fixed_base_vh_matching_brt_closed_form_with_qdot(query_position_dthetaz, robot_path=robot_path)
    n_candidates = q_candidates.shape[0]
    traj_durations = 100*np.ones(n_candidates)
    trajs = []
    throw_configs = []

    st = time.time()
    for candidate_idx in range(n_candidates):
        #return (q, phi, throw, q_dot, objLinVelInGripper, eef_velo, AE, box_position)
        throw_config_full = get_full_throwing_config_fixed_base(robot, q_candidates[candidate_idx],
                                                                phi_candidates[candidate_idx],
                                                                throw_candidates[candidate_idx],
                                                                query_position=query_position_xyz)
        assert np.linalg.norm(qdot_candidates[candidate_idx]-throw_config_full[3]) < 1e-7
        # NOTE: filter out velocity that will hit gripper palm
        # if throw_config_full[4][2] < -0.5:
        #     continue
        traj_throw = get_traj_from_ruckig(q0=q0, q0_dot=q0_dot, qd=throw_config_full[0], qd_dot=throw_config_full[3])
        traj_durations[candidate_idx] = traj_throw.duration
        trajs.append(traj_throw)
        throw_configs.append(throw_config_full)

    print("Given query (d,theta,z): ", query_position_dthetaz, ", found", len(throw_configs),
          "good throws in", "{0:0.2f}".format(1000 * (time.time() - st)), "ms")

    # Different selection rules to simulate
    selected_idx = 7000
    # mininum Ruckig duration
    # selected_idx = np.argmin(traj_durations)
    # the closest to the ball around q_throw
    # norm_cutoff = np.linalg.norm(q_candidates-q0, axis=1)-np.linalg.norm(0.4*np.ones(7))
    # norm_cutoff[norm_cutoff<0] = 100
    # selected_idx = np.argmax(norm_cutoff)

    # compute the full configuration and Ruckig trajectory of the selected throw
    #return (q, phi, throw, q_dot, objLinVelInGripper, eef_velo, AE, box_position)
    throw_config_full = get_full_throwing_config_fixed_base(robot, q_candidates[selected_idx],
                                                            phi_candidates[selected_idx],
                                                            throw_candidates[selected_idx],
                                                            query_position=query_position_xyz)
    traj_throw = get_traj_from_ruckig(q0=q0, q0_dot=q0_dot, qd=throw_config_full[0], qd_dot=throw_config_full[3])
    p.disconnect()
    video_path=robot_path+"/videos/cartesian_velocity_limit.mp4"
    # video_path=None
    # simulate Panda throw
    throw_simulation_fixed_base(traj_throw, throw_config_full, g, debug_plane=False, video_path=video_path)

    # simulate the ball only to debug Cartesian throwing configuration
    # throw_simulation_ball_only(traj_throw, throw_config_full,g, debug_plane=True)

def fixed_base_vh_matching_brt_closed_form_with_qdot(pos_target, robot_path):
    """
    VH matching for fixed base, for quick validation, we assume the inverted BRT function can be written in closed form
    :param pos_target: [d, theta, z] of the target in the base frame of the panda
    :param robot_path:
    :return:
    """
    gravity = 9.81
    # prepare grid
    robot_ts = np.load(robot_path + '/robot_rs.npy')
    robot_zs = np.load(robot_path + '/robot_zs.npy')
    gammas_deg = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    gammas_rad = np.deg2rad(gammas_deg)
    phis_deg = np.linspace(-90, 90, 13)
    phis_rad = np.deg2rad(phis_deg)

    num_robot_ts = robot_ts.shape[0]
    num_robot_zs = robot_zs.shape[0]
    num_phis = phis_deg.shape[0]
    num_gammas = gammas_deg.shape[0]

    gamma_rad_vector = np.tile(gammas_rad, [num_robot_ts, num_robot_zs, num_phis, 1])
    sin2gamma_vector = np.sin(2.0*gamma_rad_vector)
    cosgamma_square_vector = np.square(np.cos(gamma_rad_vector))

    # prepare robot data
    q_mesh = np.load(robot_path+'/qs.npy')
    translation_mesh = np.load(robot_path+'/ts.npy')
    t_mesh = np.linalg.norm(translation_mesh[:, :2], axis=1)
    beta_mesh = np.arctan2(translation_mesh[:, 1], translation_mesh[:, 0])
    z_eef_2_base_mesh = translation_mesh[:, -1]
    # correct first joint
    q_mesh_q1_corrected = q_mesh.copy()
    # now arm should be aligned with x-axis
    q_mesh_q1_corrected[:, 0] -= beta_mesh
    t_z_phi_gamma_velos_naive = np.load(robot_path+'/r_z_phi_gamma_velos_naive.npy')
    t_z_phi_gamma_q_idxs_naive = np.load(robot_path+'/r_z_phi_gamma_q_idxs_naive.npy').astype(int)
    t_z_phi_gamma_qdot = np.load(robot_path+'/r_z_phi_gamma_qdot.npy')
    # load last joint angle and objVelInGripper
    t_z_phi_gamma_q_last = np.load(robot_path + '/t_z_phi_gamma_q_last.npy')
    t_z_phi_gamma_oblVelInGripper = np.load(robot_path + '/t_z_phi_gamma_oblVelInGripper.npy')
    # for each combination of (t, \phi)
    # SSA to determine the throwing triangle
    t_vector = t_mesh[t_z_phi_gamma_q_idxs_naive]
    z_eef_2_base_vector = z_eef_2_base_mesh[t_z_phi_gamma_q_idxs_naive]
    phi_rad_vector = np.swapaxes(np.tile(phis_rad, [num_robot_ts, num_robot_zs, num_gammas, 1]), 2, 3)
    cosphi_vector = np.cos(phi_rad_vector)
    tcosphi_vector = np.multiply(t_vector, cosphi_vector)

    # start counting time from here because above matrices can be loaded offline
    st = time.time()
    # here we go to handle throwing query
    d_target = pos_target[0]
    theta_target = pos_target[1]
    z_target = pos_target[2]
    # cosine rule to determine r
    r_vector = -tcosphi_vector+np.sqrt(np.square(tcosphi_vector)-np.square(t_vector)+d_target**2)
    # to be filtered by velocity hedgehog
    r_vector = np.nan_to_num(r_vector, nan=100)
    z_vector = z_target-z_eef_2_base_vector

    s_vector = np.sqrt(gravity*np.square(r_vector)/(r_vector*sin2gamma_vector-2.0*z_vector*cosgamma_square_vector))
    # to be filtered by VH
    s_vector = np.nan_to_num(s_vector, nan=100)
    # compute the highest height
    max_height_vector = np.square(s_vector*np.sin(gamma_rad_vector))/(2.0*gravity)
    # NOTE: it's sign is defined by oy own
    psi_vector = np.arcsin(r_vector*np.abs(np.sin(phi_rad_vector))/d_target)*np.sign(phi_rad_vector)
    q1_final_vector = theta_target-psi_vector-beta_mesh[t_z_phi_gamma_q_idxs_naive]
    objVelInGripper_candidates = t_z_phi_gamma_oblVelInGripper
    # compare if such speed is feasible
    candidate_idxs = np.where((s_vector<t_z_phi_gamma_velos_naive) &
                              (s_vector<1.65) &
                              (np.abs(q1_final_vector)<(2.8973-0.4)) &
                              (max_height_vector>(z_vector-0.2)) &
                              (z_eef_2_base_vector>0.1) & # EEF higher than table
                              (t_vector<d_target) & # NOTE: otherwise, it will throw toward itself, causing large velocity that was not considered (negative phi)
                              (t_z_phi_gamma_oblVelInGripper[:,:,:,:,2]>-0.5) # objVelInGripper not too towards gripper palm
                              )
    q_candidates = q_mesh_q1_corrected[t_z_phi_gamma_q_idxs_naive[candidate_idxs]]
    r_candidates = -r_vector[candidate_idxs]
    z_candidates = -z_vector[candidate_idxs]
    phi_candidates = np.rad2deg(phi_rad_vector[candidate_idxs])
    rdot_candidates = s_vector[candidate_idxs]*np.cos(gamma_rad_vector[candidate_idxs])
    zdot_candidates = s_vector[candidate_idxs]*np.sin(gamma_rad_vector[candidate_idxs])
    x_candidates = np.concatenate((r_candidates.reshape((-1, 1)), z_candidates.reshape((-1, 1)),
                                   rdot_candidates.reshape((-1, 1)), zdot_candidates.reshape((-1, 1))), axis=-1)
    q_candidates[:, 0] = q1_final_vector[candidate_idxs]
    qlast_candidates = t_z_phi_gamma_q_last[candidate_idxs]
    q_candidates[:, -1] = qlast_candidates

    # reshape velocity
    coeff = s_vector[candidate_idxs]/t_z_phi_gamma_velos_naive[candidate_idxs]
    qdot_candidates = np.multiply(t_z_phi_gamma_qdot[candidate_idxs] ,coeff.reshape((-1, 1)))
    print("time spent for initial guess: ", time.time()-st, phi_candidates.shape[0])
    return  q_candidates, phi_candidates, x_candidates, qdot_candidates

def get_full_throwing_config_fixed_base(robot, q, phi, throw, query_position):
    """
    Return full throwing configurations
    :param robot:
    :param q:
    :param phi:
    :param throw:       [r, z, rdot, zdot] where target box is the origin
    :return:
    """
    r_throw = throw[0]
    z_throw = throw[1]
    # TODO: check which bound it belongs
    r_dot = throw[2]
    z_dot = throw[3]

    # bullet fk
    controlled_joints = [0, 1, 2, 3, 4, 5, 6]
    p.resetJointStatesMultiDof(robot, controlled_joints, [[q0_i] for q0_i in q])
    AE =p.getLinkState(robot, 11)[0]
    q = q.tolist()
    J, _ = p.calculateJacobian(robot, 11, [0, 0, 0], q+[0.1, 0.1], [0.0]*9, [0.0]*9)
    J = np.array(J)
    J = J[:,:7]

    EB = query_position[:2]-AE[:2]
    EB_dir = EB/np.linalg.norm(EB)
    J_xyz = J[:3, :]
    J_xyz_pinv = np.linalg.pinv(J_xyz)

    eef_velo = np.array([EB_dir[0]*r_dot, EB_dir[1]*r_dot, z_dot])
    q_dot = J_xyz_pinv @ eef_velo
    # assert velocity satisfies constraints
    max_qdot = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
    # print(np.abs(q_dot))
    assert np.all(np.abs(q_dot) < max_qdot)
    # error_box_position = AE + np.array([-r_throw*EB_dir[0], -r_throw*EB_dir[1], -z_throw])
    # print(np.linalg.norm(error_box_position[:2]))
    box_position = query_position

    # rotate the last joint to align the gripper with the throwing velocity
    # from https://www.programcreek.com/python/example/122109/pybullet.getEulerFromQuaternion
    gripperState = p.getLinkState(robot, 11)
    gripperPos = gripperState[0]
    gripperOrn = gripperState[1]
    invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
    eef_velo_dir_3d = eef_velo / np.linalg.norm(eef_velo)
    rotMatGripperInWorld = np.array(p.getMatrixFromQuaternion(gripperOrn)).reshape((3,3))
    objLinVelInGripper = np.linalg.inv(rotMatGripperInWorld)@eef_velo_dir_3d
    velo_angle_in_eef = np.arctan2(objLinVelInGripper[1], objLinVelInGripper[0])
    if (velo_angle_in_eef<0.5*math.pi) and (velo_angle_in_eef>-0.5*math.pi):
        eef_angle_near = velo_angle_in_eef
    elif velo_angle_in_eef>0.5*math.pi:
        eef_angle_near = velo_angle_in_eef - math.pi
    else:
        eef_angle_near = velo_angle_in_eef + math.pi
    q[-1] = eef_angle_near

    return (q, phi, throw, q_dot, objLinVelInGripper, eef_velo, AE, box_position)

def get_traj_from_ruckig(q0, q0_dot, qd, qd_dot):
    inp = InputParameter(7)
    inp.current_position = q0
    inp.current_velocity = q0_dot
    inp.current_acceleration = np.zeros(7)

    inp.target_position = qd
    inp.target_velocity = qd_dot
    inp.target_acceleration = np.zeros(7)
    inp.max_velocity = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
    inp.max_acceleration = np.array([15, 7.5, 10, 12.5, 15, 20, 20]) * 0.9
    inp.max_jerk = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000])*0.01

    otg = Ruckig(7)
    trajectory = Trajectory(7)
    _ = otg.calculate(inp, trajectory)
    return trajectory

def throw_simulation_fixed_base(trajectory, throw_config_full, g=-9.81, torque_control=False, debug_plane=False, video_path=None):
    box_position = throw_config_full[-1]
    clid = p.connect(p.GUI)
    # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=-40+140, cameraPitch=-5, cameraTargetPosition=[0.3, 0.3, 0.6])

    # FIXME: need high frequency
    hz = 1000
    delta_t = 1.0 / hz
    p.setGravity(0, 0, g)
    p.setTimeStep(delta_t)
    p.setRealTimeSimulation(0)

    # creat some visual shape to have intuitive feeling
    # from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/addPlanarReflection.py
    # from https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12165#top
    AE = throw_config_full[-2]
    EB = box_position - AE
    if debug_plane:
        yaw_debug_plane = np.arctan2(EB[1], EB[0])
        orient = p.getQuaternionFromEuler([90*(math.pi/180), 0.0, yaw_debug_plane])
        orient1 = p.getQuaternionFromEuler([90*(math.pi/180), 0.0, yaw_debug_plane+math.pi])
        # two planes for bi-directional view
        debug_plane = p.loadURDF("plane_transparent.urdf", basePosition=AE, baseOrientation=orient)
        debug_plane1 = p.loadURDF("plane_transparent.urdf", basePosition=AE, baseOrientation=orient1)

        group = 0  # other objects don't collide with me
        mask = 0  # don't collide with any other object
        # TODO: delete plane mesh
        # TODO: add plane orientation
        p.setCollisionFilterGroupMask(debug_plane, -1, group, mask)
        p.setCollisionFilterGroupMask(debug_plane1, -1, group, mask)
        texture_path = "descriptions/white.png"
        white = p.loadTexture(texture_path)
        p.changeVisualShape(debug_plane, -1, rgbaColor=[0,0,1,0.15], textureUniqueId=white)
        p.changeVisualShape(debug_plane1, -1, rgbaColor=[0,0,1,0.15], textureUniqueId=white)

    # 7: fixed joint on panda 8: fixed joint on hand 9: panda finger 1 10: panda finger 2 11: panda_grasptarget
    controlled_joints = [0, 1, 2, 3, 4, 5, 6]
    numJoints = len(controlled_joints)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    urdf_path = "descriptions/robot_descriptions/franka_panda_bullet/panda.urdf"
    # urdf_path = "franka_panda/panda.urdf"

    robotEndEffectorIndex = 11
    planeId = p.loadURDF("plane.urdf", [0, 0, -3.0])
    # soccerballId = p.loadURDF("soccerball.urdf", [-3.0, 0, 3], globalScaling=0.05)
    robotId = p.loadURDF(urdf_path, [0, 0, 0 + 0.6], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
    boxId = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/box.urdf",
                       box_position+[0, 0, 0.0 + 0.6],
                       globalScaling=0.5)
    tableId = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/table.urdf",
                       [0, 0, 0.0])
    info = p.getDynamicsInfo(soccerballId, -1)
    p.changeDynamics(soccerballId, -1, mass=1.0, linearDamping=0.00, angularDamping=0.00, rollingFriction=0.03,
                     spinningFriction=0.03, restitution=0.2, lateralFriction=0.03)
    p.changeDynamics(planeId, -1, restitution=0.9)
    # TODO: add load on the robot
    p.changeDynamics(robotId, 9, jointUpperLimit=100)
    p.changeDynamics(robotId, 10, jointUpperLimit=100)

    t0, tf = 0, trajectory.duration
    plan_time = tf - t0
    sample_t = np.arange(0, tf, delta_t)
    n_steps = sample_t.shape[0]
    traj_data = np.zeros([3, n_steps, 7])
    for i in range(n_steps):
        for j in range(3):
            traj_data[j, i] = trajectory.at_time(sample_t[i])[j]

    # reset the joint
    # see https://github.com/bulletphysics/bullet3/issues/2803#issuecomment-770206176
    q0 = traj_data[0, 0]
    p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in q0])
    eef_state = p.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
    p.resetBasePositionAndOrientation(soccerballId, eef_state[0], [0, 0, 0, 1])
    p.resetJointState(robotId, 9, 0.03)
    p.resetJointState(robotId, 10, 0.03)
    tt = 0
    flag = True

    if torque_control:
        # activating torque control by disable the default position/velocity controller
        p.setJointMotorControlArray(
                    bodyIndex=robotId,
                    jointIndices=controlled_joints,
                    controlMode=p.VELOCITY_CONTROL,
                    forces=np.zeros(numJoints))
        Kp = 0.0
        Kv = 0.0
    if not (video_path is None):
        logId = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4, fileName=video_path)

    while(True):
        if flag:
            ref = trajectory.at_time(tt)
            p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]], targetVelocities=[[q0_i] for q0_i in ref[1]])
        else:
            # if tt % 0.02 < 0.00001:
            #     a=2
            assert torque_control == False
            ref = trajectory.at_time(plan_time)
            p.resetJointStatesMultiDof(robotId, controlled_joints, [[q0_i] for q0_i in ref[0]])

        if tt > plan_time - 0*delta_t:
            p.resetJointState(robotId, 9, 0.05*10000)
            p.resetJointState(robotId, 10, 0.05*10000)
        else:
            # FIXME gripper is not rigid
            eef_state = p.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
            p.resetBasePositionAndOrientation(soccerballId, eef_state[0], [0, 0, 0, 1])
            p.resetBaseVelocity(soccerballId, linearVelocity=eef_state[-2])
        p.stepSimulation()
        tt = tt + delta_t
        if tt > trajectory.duration:
            flag = False
        time.sleep(delta_t)
        if tt > 4.0:
            break
    if not (video_path is None):
        p.stopStateLogging(logId)


    # record trajectory tracking error

    p.disconnect()

def throw_simulation_ball_only(trajectory, throw_config_full, g=-9.81, torque_control=False, debug_plane=False,
                                video_path=None):
    box_position = throw_config_full[-1]
    clid = p.connect(p.GUI)
    # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=-40 + 140, cameraPitch=-5,
                                 cameraTargetPosition=[0.3, 0.3, 0.0])

    # FIXME: need high frequency
    hz = 1000
    delta_t = 1.0 / hz
    p.setGravity(0, 0, g)
    p.setTimeStep(delta_t)
    p.setRealTimeSimulation(0)

    # creat some visual shape to have intuitive feeling
    # from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/addPlanarReflection.py
    # from https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12165#top
    AE = throw_config_full[-2]
    EB = box_position - AE
    if debug_plane:
        yaw_debug_plane = np.arctan2(EB[1], EB[0])
        orient = p.getQuaternionFromEuler([90 * (math.pi / 180), 0.0, yaw_debug_plane])
        orient1 = p.getQuaternionFromEuler([90 * (math.pi / 180), 0.0, yaw_debug_plane + math.pi])
        # two planes for bi-directional view
        debug_plane = p.loadURDF("plane_transparent.urdf", basePosition=AE, baseOrientation=orient)
        debug_plane1 = p.loadURDF("plane_transparent.urdf", basePosition=AE, baseOrientation=orient1)

        group = 0  # other objects don't collide with me
        mask = 0  # don't collide with any other object
        # TODO: delete plane mesh
        # TODO: add plane orientation
        p.setCollisionFilterGroupMask(debug_plane, -1, group, mask)
        p.setCollisionFilterGroupMask(debug_plane1, -1, group, mask)
        texture_path = "descriptions/white.png"
        white = p.loadTexture(texture_path)
        p.changeVisualShape(debug_plane, -1, rgbaColor=[0, 0, 1, 0.15], textureUniqueId=white)
        p.changeVisualShape(debug_plane1, -1, rgbaColor=[0, 0, 1, 0.15], textureUniqueId=white)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

    planeId = p.loadURDF("plane.urdf", [0, 0, -3.0])
    soccerballId = p.loadURDF("soccerball.urdf", [-3.0, 0, 3], globalScaling=0.05)
    boxId = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/box.urdf",
                       box_position + [0, 0, 0],
                       globalScaling=0.5)

    info = p.getDynamicsInfo(soccerballId, -1)
    p.changeDynamics(soccerballId, -1, mass=1.0, linearDamping=0.00, angularDamping=0.00, rollingFriction=0.03,
                     spinningFriction=0.03, restitution=0.2, lateralFriction=0.03)
    p.changeDynamics(planeId, -1, restitution=0.9)

    p.resetBasePositionAndOrientation(soccerballId, AE, [0, 0, 0, 1])

    tt = 0
    flag = True


    time.sleep(0.0)
    plan_time = 1.0
    if not (video_path is None):
        logId = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4, fileName=video_path)
    while (True):
        if flag:
            pass

        if tt > plan_time - 0 * delta_t:
            pass
        else:
            p.resetBasePositionAndOrientation(soccerballId, AE, [0, 0, 0, 1])
            p.resetBaseVelocity(soccerballId, linearVelocity=throw_config_full[-3])

        p.stepSimulation()
        tt = tt + delta_t
        if tt > trajectory.duration:
            flag = False
        time.sleep(delta_t)



if __name__ == '__main__':
    main()