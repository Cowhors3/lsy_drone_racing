"""Building blocks of the XJI online racing controller.

The package is split by responsibility:

* [track][lsy_drone_racing.control.xji.track]: geometric model of gates and obstacles.
* [route][lsy_drone_racing.control.xji.route]: collision-free waypoint routes on unknown tracks.
* [path][lsy_drone_racing.control.xji.path]: smooth geometric racing line through the waypoints.
* [timing][lsy_drone_racing.control.xji.timing]: speed profile and time-parametrised references.
* [mpc][lsy_drone_racing.control.xji.mpc]: preview MPC tracking the reference.
"""
