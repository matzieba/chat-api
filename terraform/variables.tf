variable "aws_region" {
  description = "AWS region to deploy resources"
  default     = "eu-north-1"  # Change as needed
}

variable "ec2_instance_type" {
  description = "EC2 instance type"
  default     = "t3.micro"  # Free tier eligible
}

variable "key_pair_name" {
         description = "Name of the SSH key pair"
         default     = "my-key-pair"
       }

variable "allowed_ip" {
  description = "IP address allowed to access the instance via SSH"
  default     = "0.0.0.0/0"  # Open to all (not recommended for production)
}

variable "db_username" {
  description = "Database admin username"
  type        = string
  default     = "postgresadmin"
}

variable "db_password" {
  description = "Database admin password"
  type        = string
}