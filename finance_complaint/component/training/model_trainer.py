import os

import pyspark

from finance_complaint.entity.schema import FinanceDataSchema
import sys
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.ml.pipeline import Pipeline, PipelineModel
from typing import List
from finance_complaint.config.spark_manager import spark_session
from finance_complaint.exception import FinanceException
from finance_complaint.logger import logger
from finance_complaint.entity.artifact_entity import DataTransformationArtifact, \
    PartialModelTrainerMetricArtifact, PartialModelTrainerRefArtifact, ModelTrainerArtifact
from finance_complaint.entity.config_entity import ModelTrainerConfig
from pyspark.sql import DataFrame
from pyspark.sql.functions import when, col
from pyspark.ml.feature import IndexToString
from pyspark.ml.classification import RandomForestClassifier
from finance_complaint.utils import get_score


class ModelTrainer:

    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig,
                 schema=FinanceDataSchema()
                 ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.schema = schema

    def get_scores(self, dataframe: DataFrame, metric_names: List[str]) -> List[tuple]:
        try:
            if metric_names is None:
                metric_names = self.model_trainer_config.metric_list

            scores: List[tuple] = []
            for metric_name in metric_names:
                score = get_score(metric_name=metric_name,
                                  # A keyword argument.
                                  dataframe=dataframe,
                                  label_col=self.schema.target_indexed_label,
                                  prediction_col=self.schema.prediction_column_name, )
                scores.append((metric_name, score))
            return scores
        except Exception as e:
            raise FinanceException(e, sys)
        
    from pyspark.sql.functions import when, col

    def handle_imbalance(self, train_dataframe: DataFrame) -> DataFrame:
        try:
            label_col = self.schema.target_indexed_label

            distribution = (train_dataframe.groupBy(label_col).count().collect())

            count_dict = {row[label_col]: row["count"] for row in distribution}

            major_class = max(count_dict,key=count_dict.get)

            minor_class = min(count_dict,key=count_dict.get)

            ratio = ( count_dict[major_class] /count_dict[minor_class] )

            logger.info(f"Major class: {major_class}, " f"Minor class: {minor_class}, " f"Ratio: {ratio}" )

            train_dataframe = train_dataframe.withColumn("classWeight",when( col(label_col) == minor_class, ratio).otherwise(1.0) )

            return train_dataframe

        except Exception as e:
            raise FinanceException(e, sys)

    def get_train_test_dataframe(self) -> List[DataFrame]:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            train_dataframe: DataFrame = spark_session.read.parquet(train_file_path)
            test_dataframe: DataFrame = spark_session.read.parquet(test_file_path)
            logger.info(f"Number of row in training dataframe: {train_dataframe.count()}")
            logger.info(f"Number of row in test dataframe: {test_dataframe.count()}")   
            print(f"Train row: {train_dataframe.count()} Test row: {test_dataframe.count()}")

            dataframes: List[DataFrame] = [train_dataframe, test_dataframe]
            return dataframes
        except Exception as e:
            raise FinanceException(e, sys)

    def get_model(self, label_indexer_model: StringIndexerModel) -> Pipeline:
        try:
            stages = []
            logger.info("Creating Random Forest Classifier class.")
            random_forest_clf = RandomForestClassifier(labelCol=self.schema.target_indexed_label,
                                                       featuresCol=self.schema.scaled_vector_input_features,weightCol="classWeight")

            logger.info("Creating Label generator")
            label_generator = IndexToString(inputCol=self.schema.prediction_column_name,
                                            outputCol=f"{self.schema.prediction_column_name}_{self.schema.target_column}",
                                            labels=label_indexer_model.labels)
            stages.append(random_forest_clf)
            stages.append(label_generator)
            pipeline = Pipeline(stages=stages)
            return pipeline
        except Exception as e:
            raise FinanceException(e, sys)

    def export_trained_model(self, model: PipelineModel) -> PartialModelTrainerRefArtifact:
        try:

            transformed_pipeline_file_path = self.data_transformation_artifact.exported_pipeline_file_path
            transformed_pipeline = PipelineModel.load(transformed_pipeline_file_path)

            updated_stages = transformed_pipeline.stages + model.stages
            transformed_pipeline.stages = updated_stages
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            transformed_pipeline.save(trained_model_file_path)

            logger.info("Creating trained model directory")
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            os.makedirs(os.path.dirname(trained_model_file_path), exist_ok=True)

            ref_artifact = PartialModelTrainerRefArtifact(
                trained_model_file_path=trained_model_file_path,
                label_indexer_model_file_path=self.model_trainer_config.label_indexer_model_dir)

            logger.info(f"Model trainer reference artifact: {ref_artifact}")
            return ref_artifact

        except Exception as e:
            raise FinanceException(e, sys)

    def initiate_model_training(self) -> ModelTrainerArtifact:

        try:
            dataframes = self.get_train_test_dataframe()
            train_dataframe, test_dataframe = dataframes[0], dataframes[1]


            print(f"Train row: {train_dataframe.count()} Test row: {test_dataframe.count()}")
            label_indexer = StringIndexer(inputCol=self.schema.target_column,
                                          outputCol=self.schema.target_indexed_label)
            label_indexer_model = label_indexer.fit(train_dataframe)

            os.makedirs(os.path.dirname(self.model_trainer_config.label_indexer_model_dir), exist_ok=True)
            label_indexer_model.save(self.model_trainer_config.label_indexer_model_dir)

            train_dataframe = label_indexer_model.transform(train_dataframe)
            test_dataframe = label_indexer_model.transform(test_dataframe)
            train_dataframe=self.handle_imbalance(train_dataframe)
            logger.info(f"train_dataframe after handling imbalance: {train_dataframe.count()}")
            model = self.get_model(label_indexer_model=label_indexer_model)

            trained_model = model.fit(train_dataframe)
            train_dataframe_pred = trained_model.transform(train_dataframe)
            test_dataframe_pred = trained_model.transform(test_dataframe)

            print(f"number of row in training: {train_dataframe_pred.count()}")
            scores = self.get_scores(dataframe=train_dataframe_pred,metric_names=self.model_trainer_config.metric_list)
            logger.info(f"Model trainer train metric: {scores}")
            train_metric_artifact = PartialModelTrainerMetricArtifact(areaUnderROC=scores[0][1],
                                                                      areaUnderPR=scores[1][1])
            logger.info(f"Model trainer train metric: {train_metric_artifact}")

            print(f"number of row in training: {test_dataframe_pred.count()}")
            scores = self.get_scores(dataframe=test_dataframe_pred,metric_names=self.model_trainer_config.metric_list)
            logger.info(f"Model trainer test metric: {scores}")
            test_metric_artifact = PartialModelTrainerMetricArtifact(areaUnderROC=scores[0][1],
                                                                     areaUnderPR=scores[1][1])

            logger.info(f"Model trainer test metric: {test_metric_artifact}")
            ref_artifact = self.export_trained_model(model=trained_model)

            model_trainer_artifact = ModelTrainerArtifact(model_trainer_ref_artifact=ref_artifact,
                                                          model_trainer_train_metric_artifact=train_metric_artifact,
                                                          model_trainer_test_metric_artifact=test_metric_artifact)

            logger.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise FinanceException(e, sys)
