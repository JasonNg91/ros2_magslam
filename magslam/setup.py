from setuptools import setup
import os
from glob import glob

package_name = 'magslam'
gp = "magslam/gp"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, gp],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thijs',
    maintainer_email='thijs.hof@home.nl',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
		'magmap = magslam.magmap:main',
		'magmap_control = magslam.magmap_control:main',
		'csv_feed = magslam.csv_feed:main',
		'magview = magslam.magview:main'
        ],
    },
)
