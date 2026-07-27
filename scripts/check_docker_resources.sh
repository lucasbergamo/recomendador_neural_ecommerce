#!/usr/bin/env bash
# Verifica RAM/disco disponíveis pro Docker ANTES de builds pesados (PyTorch é grande).
# Objetivo: avisar com clareza em vez de deixar a máquina travar no meio do build.
set -euo pipefail

MIN_MEMORY_GB=6
MIN_DISK_GB=10

echo "=== Verificação de recursos antes do build Docker ==="

if ! docker info >/dev/null 2>&1; then
  echo "[FALHA] Docker não está rodando. Abra o Docker Desktop e tente de novo."
  exit 1
fi

mem_bytes=$(docker info --format '{{.MemTotal}}')
mem_gb=$((mem_bytes / 1024 / 1024 / 1024))

if [ "$mem_gb" -lt "$MIN_MEMORY_GB" ]; then
  echo "[AVISO] Docker só enxerga ${mem_gb}GB de RAM (recomendado: ${MIN_MEMORY_GB}GB+)."
  echo "        O build do PyTorch é pesado — pode travar a máquina ou ficar muito lento."
  echo "        Windows/WSL2: aumente 'memory=' no .wslconfig, deixando folga pro host."
  echo "        Docker Desktop: Settings > Resources > Memory."
else
  echo "[OK]    Docker vê ${mem_gb}GB de RAM disponível."
fi

disk_avail_kb=$(df -Pk . | tail -1 | awk '{print $4}')
disk_avail_gb=$((disk_avail_kb / 1024 / 1024))

if [ "$disk_avail_gb" -lt "$MIN_DISK_GB" ]; then
  echo "[AVISO] Só ${disk_avail_gb}GB livres em disco (recomendado: ${MIN_DISK_GB}GB+)."
  echo "        Imagens com PyTorch ocupam vários GB — libere espaço antes de buildar."
else
  echo "[OK]    ${disk_avail_gb}GB livres em disco."
fi

echo "=== Fim da verificação ==="
