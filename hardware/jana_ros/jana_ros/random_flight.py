#!/usr/bin/python3
import numpy as np
from crazyflie_py import *
import threading
import time

import cffirmware as firm


CONTROLLER = 6
MODE = "fw"
SPEED = [1.0, 1.5]
DLIMITS = [1.0, 1.0]
BBOX = [[-0.5,-0.5,0.1],
        [0.5,0.5,1.0]]
DURATION = 10


def move(cf, last_pos, total_time, bbox_min, bbox_max):
    start = time.time()
    done = False
    while not done:
        if time.time() < start + total_time:
            pos = np.random.uniform(bbox_min, bbox_max)
            yaw = 0 #np.random.uniform(-np.pi, np.pi)
        else:
            # move back to the initial position at the end
            pos = np.array(cf.initialPosition) + np.array([0, 0, 0.5])
            yaw = 0
            done = True

        dist = np.linalg.norm(last_pos - pos)
        speed = np.random.uniform(SPEED[0], SPEED[1]) # m/s
        time_to_move = max(dist / speed, 2.0)
        cf.goTo(pos, yaw, time_to_move)
        if done:
            time_to_sleep = time_to_move + 0.5
        else:
            time_to_sleep = np.random.uniform(1.0, time_to_move)
        time.sleep(time_to_sleep)
        last_pos = pos

def move_lowlevel(cf, last_pos, total_time, bbox_min, bbox_max):

    planner = firm.planner()
    firm.plan_init(planner)

    e = firm.traj_eval_zero()
    e.pos.x = last_pos[0]
    e.pos.y = last_pos[1]
    e.pos.z = last_pos[2]

    start = time.time()
    done = False
    while not done:
        t = time.time() - start
        if t < total_time:
            pos = np.random.uniform(bbox_min, bbox_max)
            yaw = 0 #np.random.uniform(-np.pi, np.pi)
        else:
            # move back to the initial position at the end
            pos = np.array(cf.initialPosition) + np.array([0, 0, 0.5])
            yaw = 0
            done = True

        dist = np.linalg.norm(last_pos - pos)
        speed = np.random.uniform(SPEED[0], SPEED[1]) # m/s
        time_to_move = max(dist / speed, 2.0)
        
        #cf.goTo(pos, yaw, time_to_move)
        firm.plan_go_to_from(planner, e, False, firm.mkvec(*pos), yaw, time_to_move, t)
        planner.trajectory.timescale = 1.0
        while True:
            # compute some stats!
            vels = []
            accs = []
            for stat_t in np.arange(t, t+time_to_move, 0.01):
                e = firm.plan_current_goal(planner, stat_t)
                vels.append([e.vel.x, e.vel.y, e.vel.z])
                accs.append([e.acc.x, e.acc.y, e.acc.z])
            vels = np.array(vels)
            accs = np.array(accs)
            max_vel = np.max(np.linalg.norm(vels, axis=1))
            max_acc = np.max(np.linalg.norm(accs, axis=1))
            print(max_vel, max_acc, planner.trajectory.timescale)
            if max_vel < DLIMITS[0] and max_acc < DLIMITS[1]:
                break
            planner.trajectory.timescale *= 1.1
            time_to_move *= 1.1

        if done:
            duration_to_follow = time_to_move + 1.0
        else:
            duration_to_follow = np.random.uniform(1.0, time_to_move)
        time_to_follow = duration_to_follow + t

        #time.sleep(time_to_sleep)
        while time.time() - start < time_to_follow:

            e = firm.plan_current_goal(planner, time.time() - start)
            cf.cmdFullState(
                e.pos,
                e.vel,
                e.acc,
                e.yaw,
                e.omega)
            # print(e.pos.x, e.pos.y, e.pos.z)

            time.sleep(0.01)
        
        last_pos = pos

    cf.notifySetpointsStop()


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    allcfs.setParam("stabilizer.controller", CONTROLLER) # switch to SJC controller

    # allcfs.setParam("usec.reset", 1)

    # for the flight part
    allcfs.setParam("motorPowerSet.enable", 0) # make sure mocap can see us
    timeHelper.sleep(0.5)
    allcfs.takeoff(targetHeight=0.5, duration=3.0)
    timeHelper.sleep(3.0)

    # start recording to sdcard
    allcfs.setParam("usd.logging", 1)

    # # start thread for each cf
    # threads = []
    # total_time = DURATION
    # for _, cf in allcfs.crazyfliesById.items():
    #     cf_bbox_min = np.array(BBOX[0])
    #     cf_bbox_max = np.array(BBOX[1])
    #     pos = np.array(cf.initialPosition) + np.array([0, 0, 0.5])
    #     thread = threading.Thread(target=move_lowlevel, args=(cf, pos, total_time, cf_bbox_min, cf_bbox_max))
    #     thread.start()
    #     threads.append(thread)

    # # wait for all threads to be done
    # for thread in threads:
    #     thread.join()

    cf = allcfs.crazyflies[0]
    pos = np.array(cf.initialPosition) + np.array([0, 0, 0.5])
    if MODE == "fw":
        move(cf, pos, DURATION, BBOX[0], BBOX[1])
    else:
        move_lowlevel(cf, pos, DURATION, BBOX[0], BBOX[1])

    # stop recording to sdcard
    allcfs.setParam("usd.logging", 0)

    allcfs.land(targetHeight=0.02, duration=3.0)
    timeHelper.sleep(3.0)


if __name__ == "__main__":
    main()