#!/bin/sh
# Note: aws cloud-init requires the shebang to be `#!/bin/sh` nothing else will work!


################################### User Settings ##############################

# Define user variables 
USER="ec2-user"
HOME="/home/ec2-user"
CONDA="${HOME}/miniconda3/bin/conda"
CONDA_ACTIVATE="${HOME}/miniconda3/bin/activate"
PROJECT="https://github.com/mapa17/onomatico.git"
PROJECT_HOME="${HOME}/onomatico"

################################### Logging ####################################

# Create a log message that will be visible in /var/log/cloud-init-output.log to verify the exeuction of this script
echo "## Executing instance configuration script!"

# Terminate script on any error
set -e

# Log all output to file
exec > /tmp/setup_instance.log                                                                      
exec 2>&1

######################## Setup conda and project dependencies ##################

# Download miniconda
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh

# Install as ec2-user miniconda in batch mode
sudo -u ${USER} -- bash -c "cd ${HOME} && echo -n 'Running as ' && who && bash /tmp/miniconda.sh -b && ${CONDA} init bash"

# Create env with support for poetry (will download a lot of other stuff)
sudo -u ${USER} -- bash -c "cd ${HOME} && ${CONDA} create -n mldev poetry -y"

# Clone the repo and install dependencies
sudo -u ${USER} -- bash -c "cd ${HOME} && source ${CONDA_ACTIVATE} mldev && git clone ${PROJECT} ${PROJECT_HOME} && cd ${PROJECT_HOME} && poetry install" 
