FROM osrf/ros:humble-desktop-full-jammy

ARG ROS_DISTRO=humble

WORKDIR /gello

# Group apt updates and installs together
RUN apt update && apt install -y \
    tmux \
    libhidapi-dev \
    python3-pip \
    android-tools-adb \
    libegl1-mesa-dev && \
    rm -rf /var/lib/apt/lists/* 

# Python alias setup
RUN echo "alias python=python3" >> ~/.bashrc

# Install Python dependencies
COPY requirements.txt /gello
RUN pip install -r requirements.txt