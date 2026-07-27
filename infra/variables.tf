variable "aws_region" {
  description = "Região AWS onde tudo é provisionado"
  type        = string
  default     = "us-east-1"
}

variable "availability_zone" {
  description = "AZ única (sem multi-AZ, por decisão de escopo)"
  type        = string
  default     = "us-east-1a"
}

variable "private_subnet_cidr" {
  description = "Bloco livre na VPC default (as 6 subnets existentes vão só até 172.31.95.255)"
  type        = string
  default     = "172.31.100.0/24"
}

variable "lab_role_name" {
  description = "Role pré-criada pela AWS Academy Learner Lab — usada como execution/task role do ECS (não temos permissão pra criar roles novas)"
  type        = string
  default     = "LabRole"
}

variable "ecr_repository_name" {
  description = "Nome do repositório ECR onde a imagem 'serve' é publicada"
  type        = string
  default     = "ncf-recommender-api"
}

variable "container_port" {
  description = "Porta que o uvicorn expõe dentro do container"
  type        = number
  default     = 8000
}

variable "api_throttle_rate_limit" {
  description = "Requisições/segundo permitidas por chave de API (proteção contra uso descontrolado)"
  type        = number
  default     = 5
}

variable "api_throttle_burst_limit" {
  description = "Rajada máxima permitida acima do rate limit"
  type        = number
  default     = 10
}
