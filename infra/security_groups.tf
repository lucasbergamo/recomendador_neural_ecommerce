# SG do container ECS: só aceita tráfego na porta da API, vindo de dentro
# da própria subnet privada (é lá que o NLB também mora). Nada de fora chega aqui.
resource "aws_security_group" "ecs_task" {
  name        = "ncf-recommender-ecs-task"
  description = "Trafego para o container da API - so de dentro da subnet privada"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "API HTTP a partir do NLB (mesma subnet)"
    from_port   = var.container_port
    to_port     = var.container_port
    protocol    = "tcp"
    cidr_blocks = [var.private_subnet_cidr]
  }

  egress {
    description = "Saida liberada - restrita na pratica pela ausencia de rota de internet"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ncf-recommender-ecs-task"
  }
}

# SG dos VPC Endpoints: só aceita HTTPS (443) de dentro da subnet privada —
# é por essa porta que o container fala com ECR/CloudWatch sem sair pra internet.
resource "aws_security_group" "vpc_endpoints" {
  name        = "ncf-recommender-vpc-endpoints"
  description = "HTTPS a partir da subnet privada, para os VPC Endpoints"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "HTTPS a partir da subnet privada"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.private_subnet_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ncf-recommender-vpc-endpoints"
  }
}
