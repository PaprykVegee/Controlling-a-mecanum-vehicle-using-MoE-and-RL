#!/usr/bin/env python3
import sys
import csv

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


def main():
    if len(sys.argv) < 3:
        print("Użycie:")
        print("  python3 extract_xy_from_bag.py <bag_folder> <out_xy.txt> [topic] [frame_contains]")
        print("")
        print("Przykłady:")
        print("  # Odometry:")
        print("  python3 extract_xy_from_bag.py ~/bags/run1 ~/xy.txt /model/vehicle_blue/odometry")
        print("")
        print("  # TFMessage (pose publisher bridged jako TFMessage):")
        print("  python3 extract_xy_from_bag.py ~/bags/run1 ~/xy.txt /model/vehicle_blue/pose vehicle_blue")
        sys.exit(1)

    bag_folder = sys.argv[1]
    out_path = sys.argv[2]
    topic_name = sys.argv[3] if len(sys.argv) >= 4 else "/model/vehicle_blue/odometry"
    frame_contains = sys.argv[4] if len(sys.argv) >= 5 else "vehicle_blue"

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

    msg_type_str = type_map[topic_name]
    msg_type = get_message(msg_type_str)

    count = 0
    skipped = 0

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter=" ")

        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic != topic_name:
                continue

            msg = deserialize_message(data, msg_type)

            # Case 1: nav_msgs/msg/Odometry
            if msg_type_str.endswith("nav_msgs/msg/Odometry"):
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                w.writerow([x, y])
                count += 1
                continue

            # Case 2: tf2_msgs/msg/TFMessage
            if msg_type_str.endswith("tf2_msgs/msg/TFMessage"):
                found = False
                for tr in msg.transforms:
                    # Szukamy po child_frame_id (najczęściej tam jest nazwa modelu)
                    if frame_contains and (frame_contains not in tr.child_frame_id):
                        continue

                    x = tr.transform.translation.x
                    y = tr.transform.translation.y
                    w.writerow([x, y])
                    count += 1
                    found = True
                    break

                if not found:
                    skipped += 1
                continue

            # Inne typy – nieobsługiwane
            skipped += 1

    print(f"OK: zapisano {count} punktów do {out_path}")
    if skipped:
        print(f"INFO: pominięto {skipped} wiadomości (brak pasującego frame / nieobsługiwany format).")


if __name__ == "__main__":
    main()
