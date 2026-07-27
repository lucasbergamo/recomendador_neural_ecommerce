output "api_invoke_url" {
  description = "URL pública da API — usar com o header x-api-key"
  value       = aws_api_gateway_stage.prod.invoke_url
}

output "api_key_value" {
  description = "Valor da API key — passar no header x-api-key"
  value       = aws_api_gateway_api_key.this.value
  sensitive   = true
}

output "ecr_repository_url" {
  description = "URL do repositório ECR — usar pra docker tag/push"
  value       = aws_ecr_repository.serve.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "nlb_dns_name" {
  description = "DNS interno do NLB (só acessível de dentro da VPC)"
  value       = aws_lb.internal.dns_name
}
