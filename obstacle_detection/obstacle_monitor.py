#!/usr/bin/env python3
"""Lidar obstacle monitor for the guide-dog robot (obstacle-detection.md, sections 3-4).

Reads the merged 2D lidar scan, finds the nearest obstacle in three zones in
front of the robot (left / center / right) and publishes STOP / SLOW / CLEAR.
It only reports; it never drives the robot.

  STOP  - anything in front closer than stop_distance, or no lidar data for
          scan_timeout seconds (fail safe, never fail moving)
  SLOW  - something straight ahead closer than slow_distance
  CLEAR - otherwise

Topics published:
  /obstacle/state      std_msgs/String                 "STOP" | "SLOW" | "CLEAR"
  /obstacle/info       std_msgs/String                 JSON: state, reason, zone distances
  /obstacle/distances  std_msgs/Float32MultiArray      [left, center, right] feet (inf = nothing)
  /obstacle/markers    visualization_msgs/MarkerArray  zones for the Foxglove 3D panel

Distance parameters and outputs are in feet; scan and marker math stays in
meters (ROS convention).

Run:
  python3 obstacle_monitor.py
  python3 obstacle_monitor.py --ros-args -p stop_distance:=1.5 -p slow_distance:=5.0
"""
import json
import math

import numpy as np

M_PER_FT = 0.3048
ZONES = ('left', 'center', 'right')
COLORS = {'CLEAR': (0.1, 0.8, 0.1), 'SLOW': (1.0, 0.7, 0.0), 'STOP': (0.9, 0.1, 0.1)}


def zone_distances(ranges, angles, mount, center_half_angle, side_half_angle, min_dist):
    """Nearest obstacle distance (m, from the base frame origin) in each front zone.

    ranges/angles are the scan in the lidar frame; mount is (x, y, yaw) of the
    lidar in the base frame. Invalid readings (inf/nan) are skipped, as are hits
    closer than min_dist (the robot's own body).
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    ok = np.isfinite(ranges) & (ranges > 0.0)
    r, a = ranges[ok], angles[ok]

    tx, ty, yaw = mount
    x = tx + r * np.cos(a + yaw)
    y = ty + r * np.sin(a + yaw)
    dist = np.hypot(x, y)
    bearing = np.degrees(np.arctan2(y, x))  # 0 = straight ahead, + = left

    keep = dist >= min_dist
    dist, bearing = dist[keep], bearing[keep]

    masks = {
        'left': (bearing > center_half_angle) & (bearing <= side_half_angle),
        'center': np.abs(bearing) <= center_half_angle,
        'right': (bearing < -center_half_angle) & (bearing >= -side_half_angle),
    }
    return {z: float(dist[m].min()) if m.any() else math.inf for z, m in masks.items()}


def decide(nearest, stop_distance, slow_distance):
    """Overall (state, reason) from the per-zone nearest distances (m); reason is in feet."""
    zone = min(nearest, key=nearest.get)
    if nearest[zone] < stop_distance:
        return 'STOP', f'obstacle {nearest[zone] / M_PER_FT:.1f} ft {zone}'
    if nearest['center'] < slow_distance:
        return 'SLOW', f'obstacle {nearest["center"] / M_PER_FT:.1f} ft ahead'
    return 'CLEAR', 'path clear'


def zone_state(distance, stop_distance, slow_distance):
    if distance < stop_distance:
        return 'STOP'
    if distance < slow_distance:
        return 'SLOW'
    return 'CLEAR'


def main():
    import rclpy
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import Point
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import ColorRGBA, Float32MultiArray, String
    from tf2_ros import Buffer, TransformListener
    from visualization_msgs.msg import Marker, MarkerArray

    class ObstacleMonitor(Node):
        def __init__(self):
            super().__init__('obstacle_monitor')
            self.declare_parameter('scan_topic', '/scan')
            self.declare_parameter('base_frame', 'base_link')
            self.declare_parameter('stop_distance', 1.5)       # ft
            self.declare_parameter('slow_distance', 5.0)       # ft
            self.declare_parameter('center_half_angle', 20.0)  # deg, center zone is +-this
            self.declare_parameter('side_half_angle', 70.0)    # deg, left/right zones end here
            self.declare_parameter('robot_radius', 0.66)       # ft, ignore hits on the robot body
            self.declare_parameter('scan_timeout', 0.5)        # s, no scan this long -> STOP

            p = lambda name: self.get_parameter(name).value
            self.base_frame = p('base_frame')
            # distances are given in feet but kept in meters internally
            self.stop_distance = p('stop_distance') * M_PER_FT
            self.slow_distance = p('slow_distance') * M_PER_FT
            self.center_half_angle = p('center_half_angle')
            self.side_half_angle = p('side_half_angle')
            self.robot_radius = p('robot_radius') * M_PER_FT
            self.scan_timeout = p('scan_timeout')

            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.mounts = {}  # lidar frame -> (x, y, yaw) in base frame; the mount is fixed
            self.last_scan_time = None
            self.last_state = None

            self.state_pub = self.create_publisher(String, '/obstacle/state', 10)
            self.info_pub = self.create_publisher(String, '/obstacle/info', 10)
            self.dist_pub = self.create_publisher(Float32MultiArray, '/obstacle/distances', 10)
            self.marker_pub = self.create_publisher(MarkerArray, '/obstacle/markers', 10)
            self.create_subscription(LaserScan, p('scan_topic'), self.on_scan, qos_profile_sensor_data)
            self.create_timer(0.1, self.watchdog)

            self.get_logger().info(
                f'Watching {p("scan_topic")}: STOP < {p("stop_distance")} ft, '
                f'SLOW < {p("slow_distance")} ft ahead')

        def mount_of(self, frame):
            if frame == self.base_frame:
                return (0.0, 0.0, 0.0)
            if frame not in self.mounts:
                try:
                    t = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
                except Exception as e:
                    self.get_logger().warn(
                        f'No transform {frame} -> {self.base_frame} yet ({e})',
                        throttle_duration_sec=5.0)
                    return None
                q = t.transform.rotation
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                self.mounts[frame] = (t.transform.translation.x, t.transform.translation.y, yaw)
            return self.mounts[frame]

        def on_scan(self, scan):
            mount = self.mount_of(scan.header.frame_id)
            if mount is None:
                return
            angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
            ranges = np.asarray(scan.ranges, dtype=np.float64)
            ranges[(ranges < scan.range_min) | (ranges > scan.range_max)] = math.inf
            nearest = zone_distances(ranges, angles, mount, self.center_half_angle,
                                     self.side_half_angle, self.robot_radius)
            self.last_scan_time = self.get_clock().now()
            state, reason = decide(nearest, self.stop_distance, self.slow_distance)
            self.publish(state, reason, nearest)

        def watchdog(self):
            now = self.get_clock().now()
            if self.last_scan_time is None:
                reason = 'waiting for lidar data'
            elif (now - self.last_scan_time).nanoseconds * 1e-9 > self.scan_timeout:
                reason = 'lidar data stale'
            else:
                return
            self.publish('STOP', reason, {z: math.nan for z in ZONES})

        def publish(self, state, reason, nearest):
            if state != self.last_state:
                self.get_logger().info(f'{state}: {reason}')
                self.last_state = state
            self.state_pub.publish(String(data=state))
            info = {'state': state, 'reason': reason}
            info.update({z: (round(d / M_PER_FT, 2) if math.isfinite(d) else None)
                         for z, d in nearest.items()})
            self.info_pub.publish(String(data=json.dumps(info)))
            self.dist_pub.publish(Float32MultiArray(data=[float(nearest[z] / M_PER_FT) for z in ZONES]))
            self.marker_pub.publish(self.markers(state, nearest))

        def markers(self, state, nearest):
            stamp = self.get_clock().now().to_msg()
            c, s = self.center_half_angle, self.side_half_angle
            bounds = {'left': (c, s), 'center': (-c, c), 'right': (-s, -c)}
            out = MarkerArray()
            for i, zone in enumerate(ZONES):
                d = nearest[zone]
                zs = 'STOP' if math.isnan(d) else zone_state(d, self.stop_distance, self.slow_distance)
                m = Marker()
                m.header.frame_id = self.base_frame
                m.header.stamp = stamp
                m.ns, m.id = 'obstacle_zones', i
                m.type, m.action = Marker.TRIANGLE_LIST, Marker.ADD
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 1.0
                m.color = ColorRGBA(r=COLORS[zs][0], g=COLORS[zs][1], b=COLORS[zs][2], a=0.3)
                m.lifetime = Duration(sec=1)
                a0, a1 = bounds[zone]
                steps = max(2, int((a1 - a0) / 5))
                for k in range(steps):
                    t0 = math.radians(a0 + (a1 - a0) * k / steps)
                    t1 = math.radians(a0 + (a1 - a0) * (k + 1) / steps)
                    r = self.slow_distance
                    m.points += [Point(), Point(x=r * math.cos(t0), y=r * math.sin(t0)),
                                 Point(x=r * math.cos(t1), y=r * math.sin(t1))]
                out.markers.append(m)

            text = Marker()
            text.header.frame_id = self.base_frame
            text.header.stamp = stamp
            text.ns, text.id = 'obstacle_state', 0
            text.type, text.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            text.pose.position.z = 0.6
            text.pose.orientation.w = 1.0
            text.scale.z = 0.2
            text.color = ColorRGBA(r=COLORS[state][0], g=COLORS[state][1], b=COLORS[state][2], a=1.0)
            text.text = state
            text.lifetime = Duration(sec=1)
            out.markers.append(text)
            return out

    rclpy.init()
    node = ObstacleMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()