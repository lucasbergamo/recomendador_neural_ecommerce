"""Orquestra o pipeline completo de dados: download → bronze → silver → gold.

Delega pra preprocess.main() e features.main() em vez de duplicar a lógica —
é o mesmo caminho que o DVC executa em dois stages separados, só que num
comando só, pro conveniência do Docker/uso local.
"""

from src.data.features import main as run_feature_eng
from src.data.load import download_movielens
from src.data.preprocess import main as run_preprocess
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run() -> None:
    logger.info("pipeline_started")
    download_movielens()
    run_preprocess()
    run_feature_eng()
    logger.info("pipeline_finished")


if __name__ == "__main__":
    run()
