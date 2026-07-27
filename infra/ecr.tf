resource "aws_ecr_repository" "serve" {
  name                 = var.ecr_repository_name
  image_tag_mutability = "MUTABLE"

  # Sem isso, "terraform destroy" falha se houver imagem dentro do repositório —
  # forçamos a limpeza junto, já que essa imagem é só a demo do TC2.
  force_delete = true

  image_scanning_configuration {
    scan_on_push = true
  }
}
