#!/bin/bash

sudo python3 -m pip install --upgrade pip

# stack data compatible Spark
sudo pip3 install numpy==1.23.5 pandas==1.5.3 pyarrow==10.0.1

# images
sudo pip3 install pillow==9.5.0

# tensorflow CPU stable
sudo pip3 install tensorflow==2.12.0