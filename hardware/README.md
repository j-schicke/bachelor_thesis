# Hardware Set Up

* Crazyflie 2.1
    * URI: radio://0/80/2M/E7E7E7E702
* RPM deck
* uSD-card deck (with config.txt as configuration)
* total weight with battery: 34.7 grams
* Firmware: https://github.com/IMRCLab/crazyflie-firmware/tree/jana-thesis

## Experiments

* close blinds
* In Motive, use exposure 2500 us, LEDs off

```
ros2 launch jana_ros launch.py
ros2 run jana_ros traj
```

or

```
ros2 launch jana_ros launch.py
ros2 run jana_ros random_flight
```

## Data

### jana00

HEIGHT = 0.7
TIMESCALE = 1.0
CONTROLLER = 6
FILE = "figure8.csv"

### jana01

HEIGHT = 0.7
TIMESCALE = 5.0
CONTROLLER = 6
FILE = "yaw0.csv"

### jana02

HEIGHT = 0.3
TIMESCALE = 0.8
CONTROLLER = 2
FILE = "figure8.csv"

### jana03

HEIGHT = 0.3
TIMESCALE = 0.8
CONTROLLER = 1
FILE = "figure8.csv"

### jana04

HEIGHT = 0.5
TIMESCALE = 5.0
CONTROLLER = 2
FILE = "yaw0.csv"

### jana05

HEIGHT = 0.5
TIMESCALE = 4.0
CONTROLLER = 1
FILE = "yaw0.csv"

### jana06

HEIGHT = 0.5
TIMESCALE = 2.0
CONTROLLER = 1
FILE = "yaw0.csv"

### jana10

HEIGHT = 0.5
TIMESCALE = 5.0
CONTROLLER = 2
FILE = "circle0.csv"

### jana11

HEIGHT = 0.5
TIMESCALE = 3.0
CONTROLLER = 1
FILE = "circle0.csv"

### jana20

```
CONTROLLER = 2
MODE = "fw"
SPEED = [0.1, 0.5]
DLIMITS = [1.0, 1.0]
BBOX = [[-0.5,-0.5,0.1],
        [0.5,0.5,1.0]]
DURATION = 10
```

### jana23

```
CONTROLLER = 2
MODE = "fw"
SPEED = [0.1, 1.0]
DLIMITS = [1.0, 1.0]
BBOX = [[-0.5,-0.5,0.1],
        [0.5,0.5,1.0]]
DURATION = 10
```

### jana24

```
CONTROLLER = 1
MODE = "fw"
SPEED = [0.1, 1.0]
DLIMITS = [1.0, 1.0]
BBOX = [[-0.5,-0.5,0.1],
        [0.5,0.5,1.0]]
DURATION = 10
```

### jana25

```
CONTROLLER = 6
MODE = "fw"
SPEED = [0.1, 1.0]
DLIMITS = [1.0, 1.0]
BBOX = [[-0.5,-0.5,0.1],
        [0.5,0.5,1.0]]
DURATION = 10
```

### jana27

```
CONTROLLER = 2
MODE = "fw"
SPEED = [0.5, 1.0]
DLIMITS = [1.0, 1.0]
BBOX = [[-0.5,-0.5,0.1],
        [0.5,0.5,1.0]]
DURATION = 10
```

### jana28

```
CONTROLLER = 2
MODE = "fw"
SPEED = [0.5, 1.0]
DLIMITS = [1.0, 1.0]
BBOX = [[-0.5,-0.5,0.1],
        [0.5,0.5,1.0]]
DURATION = 10
```

### jana29

```
CONTROLLER = 6
MODE = "fw"
SPEED = [0.5, 1.0]
DLIMITS = [1.0, 1.0]
BBOX = [[-0.5,-0.5,0.1],
        [0.5,0.5,1.0]]
DURATION = 10
```

### jana30

```
CONTROLLER = 1
MODE = "fw"
SPEED = [1.0, 1.5]
DLIMITS = [1.0, 1.0]
BBOX = [[-0.5,-0.5,0.1],
        [0.5,0.5,1.0]]
DURATION = 10
```

### jana32

```
CONTROLLER = 2
MODE = "fw"
SPEED = [1.0, 1.5]
DLIMITS = [1.0, 1.0]
BBOX = [[-0.5,-0.5,0.1],
        [0.5,0.5,1.0]]
DURATION = 10
```

### jana33

```
CONTROLLER = 6
MODE = "fw"
SPEED = [1.0, 1.5]
DLIMITS = [1.0, 1.0]
BBOX = [[-0.5,-0.5,0.1],
        [0.5,0.5,1.0]]
DURATION = 10
```


