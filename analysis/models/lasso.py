import inspect
import os
import pandas as pd
import pickle
from datetime import datetime

from sklearn.linear_model import LogisticRegression

from analysis.models.data_pipeline import DataPipeline
from sklearn.pipeline import Pipeline
from analysis.models.evaluator import evaluate_classification
#from analysis.features.build_features import DenseFeatureTransformer
from analysis.features.build_features import FeatureBuilder, FeaturePreprocessor


def run(data_path, mode="train", save_predictions=True, save_model=True):
    fb = FeatureBuilder()

    ### TRAIN MODE
    if mode == "train":
        dp = DataPipeline(data_path, label_columns=["toxic"])
        X_train, X_test, y_train, y_test = dp.get_data()

        #y_train = y_train.ravel()
        #y_test = y_test.ravel()

        if os.path.exists(fb.tfidf_path):
            #print("Loading cached TF-IDF vectorizers...")
            fb.load()
        else:
            print("Fitting TF-IDF vectorizers...")
            fb.fit(X_train)
        
        print("Transforming train features...")
        #print("ABOUT TO CALL fb.transform(train)")
        #print("fb type:", type(fb))
        #import inspect
        #print("fb class file:", inspect.getfile(fb.__class__))
        X_train_feat = fb.transform(X_train, split="train")
        print("Transforming test features...")
        X_test_feat = fb.transform(X_test, split="test")

        preprocessor = FeaturePreprocessor()
        print("Preprocessing features...")
        X_train_proc = preprocessor.fit_transform(X_train_feat)
        X_test_proc  = preprocessor.transform(X_test_feat)


        model = LogisticRegression(
            penalty="l1",      
            solver="saga",    
            C=1.0,
            max_iter=1000,
            random_state=42
        )

        pipeline = Pipeline([
            ("model", model)
        ])

        print("Fitting pipeline...")

        y_train = y_train.values.ravel()
        pipeline.fit(X_train_proc, y_train)
        y_pred = pipeline.predict(X_test_proc)

        y_test = y_test.values.ravel()

        metrics = evaluate_classification(
            y_test,
            y_pred,
            None,
            name="Lasso Baseline (LogReg L1)"
        )

        if save_predictions:
            os.makedirs("./analysis/models/model_outputs/lasso/predictions", exist_ok=True)

            df = pd.DataFrame({ "true": y_test,"pred": y_pred })

            df.to_csv(
                "./analysis/models/model_outputs/lasso/predictions/lasso_predictions.csv",
                index=False
            )

        if save_model:
            os.makedirs("./analysis/models/artifacts", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            with open(f"./analysis/models/artifacts/lasso_{timestamp}.pkl", "wb") as f:
                pickle.dump(pipeline, f)

    elif mode == "infer":
        # load model
        model_path = "./analysis/models/artifacts/lasso_latest.pkl"
        if not os.path.exists(model_path):
            raise ValueError("No trained model found.")

        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)

        fb.load()

        df = pd.read_csv(data_path)
        X = df["comment_text"]

        dp = DataPipeline(data_path, label_columns=["toxic"])
        #X_train, X_test, y_train, y_test = dp.get_data()
        X, y_test = dp.get_infer_data(infer_path='./data/processed/test_data.pkl')

        X_feat = fb.transform(X, split="test")
        print("Preprocessing...")
        X_proc = preprocessor.transform(X_feat)
        y_pred = pipeline.predict(X_proc)
        y_test = y_test.values.ravel()

        if save_predictions:
            os.makedirs("./analysis/models/model_outputs/lasso", exist_ok=True)
            pd.DataFrame({
                "pred": y_pred
            }).to_csv("./analysis/models/model_outputs/lasso/lasso_infer.csv", index=False)
    else:
        raise ValueError("mode must be 'train' or 'infer'")