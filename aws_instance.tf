provider "aws" {
  region = "eu-central-1"
}

terraform {
  required_version = ">= 1.0.7"
}

variable "key_path" {
  default = "~/.ssh/aws_dev.pub"
}

resource "aws_key_pair" "pub-key" {
  key_name   = "pub-key"
  public_key = file("${var.key_path}")
}

resource "aws_instance" "instance" {
  ami                         = "ami-0b879110efb09396b"
  #instance_type               = "p3.2xlarge"
  instance_type               = "t2.small"
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


output "instance-public-ip" {
  value = aws_instance.instance.public_ip
}