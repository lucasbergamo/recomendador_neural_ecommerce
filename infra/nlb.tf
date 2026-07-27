# NLB interno — sem IP público. Existe só porque o VPC Link do API Gateway
# REST API (v1) exige um Network Load Balancer como alvo; não estamos
# balanceando carga entre réplicas (só existe 1 task).
resource "aws_lb" "internal" {
  name               = "ncf-recommender-internal"
  internal           = true
  load_balancer_type = "network"
  subnets            = [aws_subnet.private.id]

  tags = {
    Name = "ncf-recommender-internal"
  }
}

# target_type "ip": obrigatório para Fargate (não existe instância EC2 pra
# apontar, o alvo é o IP da ENI que o Fargate cria pra cada task).
resource "aws_lb_target_group" "serve" {
  name        = "ncf-recommender-tg"
  port        = var.container_port
  protocol    = "TCP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  health_check {
    protocol            = "HTTP"
    path                = "/health"
    port                = var.container_port
    interval            = 10
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }

  # Task Spot pode morrer a qualquer momento — desregistrar rápido evita
  # mandar tráfego pra um alvo que já era.
  deregistration_delay = 30

  tags = {
    Name = "ncf-recommender-tg"
  }
}

resource "aws_lb_listener" "serve" {
  load_balancer_arn = aws_lb.internal.arn
  port              = var.container_port
  protocol          = "TCP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.serve.arn
  }
}
