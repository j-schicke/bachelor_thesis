#!/usr/bin/python3
import numpy as np
from crazyflie_py import *
import threading
import time

CONTROLLER = 2
SPEED = [0.1, 1.5]
BBOX = [[-0.75,-0.75,0.1],
        [0.75,0.75,1.5]]
DURATION = 30


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
        time_to_sleep = np.random.uniform(1.0, time_to_move)
        time.sleep(time_to_sleep)
        last_pos = pos

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

    # start thread for each cf
    threads = []
    total_time = DURATION
    for _, cf in allcfs.crazyfliesById.items():
        cf_bbox_min = np.array(BBOX[0])
        cf_bbox_max = np.array(BBOX[1])
        pos = np.array(cf.initialPosition) + np.array([0, 0, 0.5])
        thread = threading.Thread(target=move, args=(cf, pos, total_time, cf_bbox_min, cf_bbox_max))
        thread.start()
        threads.append(thread)

    # wait for all threads to be done
    for thread in threads:
        thread.join()

    # stop recording to sdcard
    allcfs.setParam("usd.logging", 0)

    allcfs.land(targetHeight=0.02, duration=3.0)
    timeHelper.sleep(3.0)


if __name__ == "__main__":
    main()