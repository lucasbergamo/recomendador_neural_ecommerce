resource "aws_cloudwatch_log_group" "serve" {
  name              = "/ecs/ncf-recommender"
  retention_in_days = 7
}

resource "aws_ecs_cluster" "this" {
  name = "ncf-recommender-cluster"
}

# Habilita FARGATE_SPOT no cluster — sem isso, o Service não pode usar essa
# estratégia de capacidade.
resource "aws_ecs_cluster_capacity_providers" "this" {
  cluster_name = aws_ecs_cluster.this.name

  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 1
  }
}

resource "aws_ecs_task_definition" "serve" {
  family                   = "ncf-recommender-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "3072"
  execution_role_arn       = data.aws_iam_role.lab_role.arn
  task_role_arn            = data.aws_iam_role.lab_role.arn

  container_definitions = jsonencode([
    {
      name      = "ncf-recommender-api"
      image     = "${aws_ecr_repository.serve.repository_url}:latest"
      essential = true
      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.serve.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])
}

# O Service é o que dá "self-healing": mantém desired_count=1 sempre —
# se a Spot for reclamada e a task morrer, o ECS pede uma nova sozinho.
resource "aws_ecs_service" "serve" {
  name            = "ncf-recommender-service"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.serve.arn
  desired_count   = 1

  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 1
  }

  network_configuration {
    subnets          = [aws_subnet.private.id]
    security_groups  = [aws_security_group.ecs_task.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.serve.arn
    container_name   = "ncf-recommender-api"
    container_port   = var.container_port
  }

  # Tempo de graça antes do health check do NLB poder derrubar a task —
  # dá tempo do uvicorn subir e carregar o modelo (PyTorch import não é instantâneo).
  health_check_grace_period_seconds = 30

  depends_on = [
    aws_lb_listener.serve,
    aws_vpc_endpoint.ecr_api,
    aws_vpc_endpoint.ecr_dkr,
    aws_vpc_endpoint.logs,
    aws_vpc_endpoint.s3,
  ]
}
