import mlflow


class MLFlowLogger:
    def __init__(self, tracking_url: str, experiment_name: str):
        self.experiment_name = experiment_name
        self.tracking_url = tracking_url
        mlflow.set_tracking_uri(tracking_url)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()

        print(f"MLFlowLogger initialized with experiment '{experiment_name}' at '{tracking_url}'")


    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int = None):
        mlflow.log_metrics(metrics, step=step)

    def end_run(self):
        mlflow.end_run()