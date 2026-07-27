data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_iam_role" "lab_role" {
  name = var.lab_role_name
}
