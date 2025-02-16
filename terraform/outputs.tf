output "instance_public_ip" {
  description = "Public IP of the EC2 instance"
  value       = aws_instance.django_ec2.public_ip
}

output "instance_public_dns" {
  description = "Public DNS of the EC2 instance"
  value       = aws_instance.django_ec2.public_dns
}