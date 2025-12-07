#!/usr/bin/env python3
import sys
import csv

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


def main():
    if len(sys.argv) < 3:
        print("Użycie: python3 extract_xy_from_bag.py <bag_folder> <out_xy.txt> [topic]")
        print("Np:     python3 extract_xy_from_bag.py ~/ros2_ws/gt_run_01 ~/ros2_ws/xy.txt /odometry_10hz")
        sys.exit(1)

    bag_folder = sys.argv[1]
    out_path = sys.argv[2]
    topic_name = sys.argv[3] if len(sys.argv) >= 4 else "/odometry_10hz"

    storage_options = rosbag2_py.StorageOptions(uri=bag_folder, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topics}
    if topic_name not in type_map:
        print("Dostępne topiki w bagnie:")
        for t in topics:
            print(f"  {t.name}  [{t.type}]")
        raise RuntimeError(f"Nie ma topiku {topic_name} w tym bagu.")

    msg_type = get_message(type_map[topic_name])

    count = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter=" ")
        # zapis: "x y"
        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic != topic_name:
                continue
            msg = deserialize_message(data, msg_type)
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            w.writerow([x, y])
            count += 1

    print(f"OK: zapisano {count} punktów do {out_path}")


if __name__ == "__main__":
    main()
