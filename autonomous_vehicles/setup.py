from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'autonomous_vehicles'

setup(
    name=package_name,
    version='0.0.0',
    # Używamy find_packages() do automatycznego znajdowania wszystkich modułów/klas.
    packages=find_packages(exclude=['test']),
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Patryk',
    maintainer_email='you@example.com',
    description='Pakiet ROS 2 dla pojazdu autonomicznego',
    license='Apache-2.0',
    tests_require=['pytest'],
    
    # ✅ Sekcja data_files (Przyjęta bezpieczna składnia)
    data_files=[
        # 1. Instalacja resource file (Kluczowe dla ament_index)
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        
        # 2. Instalacja launch files (Jeśli masz katalog 'launch')
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        
        # 3. Instalacja config files (Jeśli masz katalog 'config')
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        
        # UWAGA: Usunięto linię instalującą package.xml, aby uniknąć błędu kompilacji!
    ],
    
    # ✅ Sekcja entry_points (Zakładamy, że pliki to np. control_node.py)
    entry_points={
        'console_scripts': [
            # Zmieniono na małe litery, aby pasowało do konwencji
            'control_node = autonomous_vehicles.control_node:main',
            'segmentation = autonomous_vehicles.segmentation:main',
            'view_man = autonomous_vehicles.view_manipulation_node:main',
            # Dodaj swoje pozostałe węzły tutaj, używając małych liter i kropki (np. .nazwa_pliku:main)
        ],
    },
)