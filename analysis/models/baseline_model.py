import os
import pandas as pd
import pickle
from sklearn.dummy import DummyClassifier

from analysis.models.data_pipeline import DataPipeline
from analysis.models.evaluator import evaluate_classification
from sklearn.pipeline import Pipeline
from analysis.features.build_features import DenseFeatureTransformer, TfidfTransformer, BertTransformer

def run(data_path, save_predictions=True, save_model=True):
    # Load the dataset
    dp = DataPipeline(data_path, label_columns="toxic")
    X_train, X_test, y_train, y_test = dp.get_data()

    model = DummyClassifier(strategy="stratified", random_state=42)

    pipeline = Pipeline([
        ("dense", DenseFeatureTransformer()),
        #("tfidf", TfidfTransformer()),
        #("bert", BertTransformer()),
        ("baseline_model", model)
    ]) 

    X_train_dummy = [[0]] * len(y_train)
    X_test_dummy = [[0]] * len(y_test)

    # Fit the pipeline on the training data
    X_train_dummy = pd.DataFrame(X_train_dummy, columns=['comment_text'])
    X_test_dummy = pd.DataFrame(X_test_dummy, columns=['comment_text'])

    pipeline.fit(X_train_dummy, y_train)
    y_pred = pipeline.predict(X_test_dummy)

    metrics = evaluate_classification(
        y_test,
        y_pred,
        name="Dummy Baseline"
    )

    if save_predictions:
        os.makedirs("./analysis/models/model_outputs/baseline/predictions", exist_ok=True)

        df = pd.DataFrame({
            #"comments": X_train["comment_text"],
            "true": y_test,
            "pred": y_pred
        })

        df.to_csv("./analysis/models/model_outputs/baseline/predictions/dummy_predictions.csv", index=False)

    if save_model:
        os.makedirs("./analysis/models/artifacts", exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(f"./analysis/models/artifacts/dummy_{timestamp}.pkl", "wb") as f:
            pickle.dump(model, f)

    return metrics

