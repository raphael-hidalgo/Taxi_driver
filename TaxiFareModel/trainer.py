from sklearn.pipeline import Pipeline
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        pipe = ColumnTransformer([('distance', dist_pipe, [
                "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
                'dropoff_longitude']), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")

        return pipe

    def run(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)
        abc = self.set_pipeline().fit_transform(X_train, y_train)
        model = LinearRegression().fit(abc, y_train)
        return model, X_test, y_test, X_train, y_train

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        model, X_test, y_test, X_train, y_train = self.run()
        pipe = self.set_pipeline().fit(X_train, y_train)
        y = pipe.transform(X_test)
        y_pred = model.predict(y)
        return compute_rmse(y_pred, y_test)

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    y = df.pop('fare_amount')
    X = df
    data = Trainer(X, y)
    print(data.evaluate())
