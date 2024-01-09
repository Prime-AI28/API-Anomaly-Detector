#!/bin/bash


# Function to check and install packages
check_install_package() {
    PACKAGE=$1
    RETRY_COUNT=3
    CURRENT_TRY=0

    while [ $CURRENT_TRY -lt $RETRY_COUNT ]; do
        apt-get install -y $PACKAGE
        if [ $? -eq 0 ]; then
            echo "$PACKAGE installed successfully."
            break
        else
            echo "Failed to install $PACKAGE. Retrying..."
            let CURRENT_TRY++
            sleep 1
        fi
    done

    if [ $CURRENT_TRY -eq $RETRY_COUNT ]; then
        echo "Failed to install $PACKAGE after $RETRY_COUNT attempts. Exiting."
        exit 1
    fi
}

# Update package lists and install required dependencies
apt-get update
check_install_package "apt-transport-https"
check_install_package "ca-certificates"
check_install_package "curl"

# Add Kubernetes repository and install necessary components
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.28/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.28/deb/ /' | tee /etc/apt/sources.list.d/kubernetes.list
apt-get update
check_install_package "kubectl"
check_install_package "kubeadm"
check_install_package "kubelet"
check_install_package "kubernetes-cni"

# Configure sysctl settings
echo 'net.bridge.bridge-nf-call-iptables = 1' >> /etc/sysctl.conf
echo 'net.ipv4.ip_forward = 1' >> /etc/sysctl.conf
echo 'net.bridge.bridge-nf-call-ip6tables = 1' >> /etc/sysctl.conf
sysctl -p

modprobe overlay
modprobe br_netfilter 

# Install necessary tools and dependencies for CRI-O
check_install_package "policycoreutils"
sestatus
check_install_package "podman"

export OS=xUbuntu_22.04
export VERSION=1.26
apt-get update

# Add CRI-O repository and install CRI-O
echo "deb https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/$OS/ /" > /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list
echo "deb http://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable:/cri-o:/$VERSION/$OS/ /" > /etc/apt/sources.list.d/devel:kubic:libcontainers:stable:cri-o:$VERSION.list
curl -L https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable:/cri-o:/$VERSION/$OS/Release.key | apt-key add -
curl -L https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/$OS/Release.key | apt-key add -
apt-get update
check_install_package "cri-o"
check_install_package "cri-o-runc"

# Configure CRI-O
sed -i 's/# network_dir/network_dir/' /etc/crio/crio.conf
sed -i '/# plugin_dirs = \[/,/# \]/ s/# //' /etc/crio/crio.conf
sed -i 's/10.85.0.0/10.244.0.0/' /etc/cni/net.d/100-crio-bridge.conflist

apt-get upgrade -y cri-o cri-o-runc


# Start and enable CRI-O
systemctl start crio
systemctl enable crio

# Disable swap
swapoff -a
sed -i '/\/swapfile/s/^/#/' /etc/fstab
