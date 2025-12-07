from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'autonomous_vehicles'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Patryk',
    maintainer_email='you@example.com',
    description='Pakiet ROS 2 dla pojazdu autonomicznego',
    license='Apache-2.0',
    tests_require=['pytest'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    entry_points={
        'console_scripts': [
            'control_node = autonomous_vehicles.control_node:main',
            'image_node = autonomous_vehicles.image_node:main',
            'lidar_node = autonomous_vehicles.lidar_node:main',
            'lidar_2d_node = autonomous_vehicles.lidar_2d_node:main',
            'offroad_checker = autonomous_vehicles.offroad_checker:main',
            'gps_node = autonomous_vehicles.gps_node:main'

        ],
    },
)
