# Subnet privada: SEM rota pra Internet Gateway. O único jeito de sair dela
# é pelos VPC Endpoints (vpc_endpoints.tf) — ECR e CloudWatch Logs, nada mais.
resource "aws_subnet" "private" {
  vpc_id            = data.aws_vpc.default.id
  cidr_block        = var.private_subnet_cidr
  availability_zone = var.availability_zone

  tags = {
    Name = "ncf-recommender-private"
  }
}

# Tabela de rotas isolada: só a rota "local" implícita da VPC existe.
# Nenhuma rota 0.0.0.0/0 — é isso que torna a subnet "privada" de verdade,
# não apenas um nome.
resource "aws_route_table" "private" {
  vpc_id = data.aws_vpc.default.id

  tags = {
    Name = "ncf-recommender-private-rt"
  }
}

resource "aws_route_table_association" "private" {
  subnet_id      = aws_subnet.private.id
  route_table_id = aws_route_table.private.id
}
