################################ User variables ################################

# Path to private and public key used with AWS
variable "prv_key_path" {
  default = "~/.ssh/aws_dev"
}

variable "pub_key_path" {
  default = "~/.ssh/aws_dev.pub"
}

# AWS Instance type that should be deployed
variable "instance_type"{
  #default = "p2.xlarge"
  default = "t2.small"
}

# AWS AMI Image ID
variable "ami"{
  default = "ami-0b879110efb09396b"
}

# AWS Region we want the instance to run in
provider "aws" {
  region = "eu-central-1"
}


########################### Terraform Configuration ############################

terraform {
  required_version = ">= 1.0.7"
}

resource "aws_key_pair" "pub-key" {
  key_name   = "pub-key"
  public_key = file("${var.pub_key_path}")
}

# Define an AWS instance using the usr variables from above
# Make user of the user_data (ie.e cloud init) option to define a bash script
# to configure the instance at the first boot
resource "aws_instance" "instance" {
  ami                         = "${var.ami}"
  instance_type               = "${var.instance_type}"
  key_name                    = aws_key_pair.pub-key.id
  vpc_security_group_ids      = [aws_security_group.sg.id]
  associate_public_ip_address = true
  user_data = "${file("setup_instance.sh")}"

  root_block_device {
    volume_size           = 60
    delete_on_termination = true
  }

  tags = {
    Name = "AWS_ML_Instance"
  }
}

# Our user_data script will be executed as part of cloud-init final
# Block terraform until cloud-init has finished executing our script and the instance is ready.
# Create a special null_resource that will use remote-exec to wait until cloud-init has finished
resource "null_resource" "cloud_init_wait" {
  connection {
    host        = "${aws_instance.instance.public_ip}"
    user        = "ec2-user"
    private_key = "${file("${var.prv_key_path}")}"
    #script_path = "/tmp/cloud_init_wait.sh"
    timeout     = "10m"
  }
  provisioner "remote-exec" {
    inline = ["sudo cloud-init status --wait"]
  }
  depends_on = [aws_instance.instance]
}

# Define a basic security group that restricts inbound access to ssh but allows
# all outgoing access
resource "aws_security_group" "sg" {
  name        = "my_security_group"
  description = "Only allow inbound ssh access"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }


  tags = {
    Name = "AWS_ML_DEV_INSTANCE"
  }
}

# On termination of the script set an output varialbe containing the public ip
# of the instance so we can access it using ssh
output "instance-public-ip" {
  value = aws_instance.instance.public_ip
}