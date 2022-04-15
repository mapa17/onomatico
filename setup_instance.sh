#!/bin/sh

echo "#### MY SCRIPT !!! ###"

# Log all to file
exec > /tmp/setup_instance.log                                                                      
exec 2>&1

# Download miniconda
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh

# Install as ec2-user miniconda in batch mode
sudo -u ec2-user -- bash -c 'cd /home/ec2-user && echo -n "Running as " && who && bash /tmp/miniconda.sh -b && ./miniconda/bin/conda init bash'

# Create env with support for poetry (will download a lot of other stuff)
sudo -u ec2-user -- bash -c 'cd /home/ec2-user && ./miniconda3/bin/conda create -n mldev poetry -y'
