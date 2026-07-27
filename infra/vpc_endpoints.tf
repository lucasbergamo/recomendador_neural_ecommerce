# VPC Endpoints: a subnet privada não tem rota pra internet, então o container
# não consegue "sair" pra falar com ECR/CloudWatch do jeito normal. Os endpoints
# criam um caminho direto (PrivateLink) até esses serviços AWS, sem sair da rede
# da AWS — dispensa NAT Gateway (que exigiria Elastic IP, limitado em Learner Labs).

# Tipo "Interface": cria um ENI dentro da subnet, com IP privado.
resource "aws_vpc_endpoint" "ecr_api" {
  vpc_id              = data.aws_vpc.default.id
  service_name        = "com.amazonaws.${var.aws_region}.ecr.api"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = [aws_subnet.private.id]
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true
}

resource "aws_vpc_endpoint" "ecr_dkr" {
  vpc_id              = data.aws_vpc.default.id
  service_name        = "com.amazonaws.${var.aws_region}.ecr.dkr"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = [aws_subnet.private.id]
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true
}

resource "aws_vpc_endpoint" "logs" {
  vpc_id              = data.aws_vpc.default.id
  service_name        = "com.amazonaws.${var.aws_region}.logs"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = [aws_subnet.private.id]
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true
}

# Tipo "Gateway": não usa ENI/subnet — anexa direto na tabela de rotas.
# ECR guarda as camadas de imagem (layers) no S3 por baixo dos panos,
# então esse endpoint é necessário mesmo sem "usar S3" diretamente no código.
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = data.aws_vpc.default.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]
}
