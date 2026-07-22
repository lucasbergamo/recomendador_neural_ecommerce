"""Registra o NCF final no MLflow Model Registry e promove via aliases.

MLflow >= 2.9 descontinuou os "stages" antigos do Model Registry
(Staging/Production/Archived como campos nativos da versão). A recomendação
oficial é usar aliases arbitrários por versão (`@staging`, `@production`,
`@champion`, etc.) — são só tags mutáveis, não estados do ciclo de vida do
MLflow, então a promoção é literalmente reatribuir o alias pra outra versão.
"""

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

from src.data.features import load_gold
from src.models.factory import RecommenderFactory
from src.utils.config import MODELS_DIR, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "ncf-recommender"


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    _, _, _, meta = load_gold()
    n_users, n_items = int(meta["n_users"]), int(meta["n_items"])

    model = RecommenderFactory.create_neural(n_users=n_users, n_items=n_items)
    model.load(MODELS_DIR / "ncf.pt")

    with mlflow.start_run(run_name="register_ncf_recommender"):
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
        )

    version = model_info.registered_model_version
    client = MlflowClient()

    client.set_registered_model_alias(MODEL_NAME, "staging", version)
    logger.info("model_promoted", alias="staging", version=version)

    client.set_registered_model_alias(MODEL_NAME, "production", version)
    logger.info("model_promoted", alias="production", version=version)


if __name__ == "__main__":
    main()
