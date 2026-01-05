# from setuptools import setup
# import os
# from glob import glob

# package_name = 'autonomous_vehicles'

# setup(
#     # ... inne pola
#     packages=[package_name],  # To jest kluczowe! Musi pasować do nazwy folderu
#     data_files=[
#         # Dodaj pliki launch i konfiguracyjne, jeśli ich używasz
#         (os.path.join('share', package_name), glob('package.xml')),
#         # Instalacja plików launch (np. launch/launch.py)
#         (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[py]*'))), 
#         # Instalacja plików konfiguracyjnych (np. config/bridge_config.yaml)
#         (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))), 
#     ],
#     # ... inne pola
#     entry_points={
#         'console_scripts': [
#             # Upewnij się, że węzły są tu zadeklarowane
#             'control_node = autonomous_vehicles.ControlNode:main',
#             'segmentation = autonomous_vehicles.Segmetation:main', # Popraw literówkę Segmetation -> Segmentation jeśli to zamierzone
#             'view_manipulation = autonomous_vehicles.view_manipualtion:main', # Popraw literówkę manipualtion -> manipulation
#         ],
#     },
# )