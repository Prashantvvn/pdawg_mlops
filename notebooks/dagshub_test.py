import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/Prashantvvn/pdawg_mlops.mlflow")

dagshub.init(repo_owner='Prashantvvn', repo_name='pdawg_mlops', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)