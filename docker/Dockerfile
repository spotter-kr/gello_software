FROM osrf/ros:humble-desktop-full-jammy

ARG ROS_DISTRO=humble

# Group apt updates and installs together
RUN apt update && apt install -y \
    tmux \
    libhidapi-dev \
    python3-pip \
    android-tools-adb \
    libegl1-mesa-dev && \
    rm -rf /var/lib/apt/lists/* 

# Install Python dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt
    # && pip install -e /gello \
    # && pip install -e /gello/third_party/DynamixelSDK/python

# Add a non-root user 'spotter' with sudo permission
ARG USER
ARG UID
ARG GID=${UID}

RUN groupadd -g ${GID} ${USER} \
    && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USER} \
    && usermod -a -G dialout ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers \
    && chmod 0440 /etc/sudoers

# Python alias setup
RUN echo "alias python=python3" >> /home/${USER}/.bashrc \
    && echo ". /opt/ros/$ROS_DISTRO/setup.bash" >> /home/${USER}/.bashrc \
    && echo "sudo pip install -e /home/${USER}/workspace" >> /home/${USER}/.bashrc \
    && echo "sudo pip install -e /home/${USER}/workspace/third_party/DynamixelSDK/python" >> /home/${USER}/.bashrc
    