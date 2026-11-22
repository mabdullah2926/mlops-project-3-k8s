import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from kfp import dsl, compiler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@dsl.component(base_image="python:3.12-slim", packages_to_install=["scikit-learn", "joblib"])
def load_and_preprocess_data(
    X_train_path: dsl.OutputPath(str),
    X_test_path: dsl.OutputPath(str),
    y_train_path: dsl.OutputPath(str),
    y_test_path: dsl.OutputPath(str),
):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import joblib

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    joblib.dump(X_train, X_train_path)
    joblib.dump(X_test, X_test_path)
    joblib.dump(y_train, y_train_path)
    joblib.dump(y_test, y_test_path)


@dsl.component(base_image="python:3.12-slim", packages_to_install=["scikit-learn", "joblib"])
def train_model(
    X_train_path: dsl.InputPath(str),
    y_train_path: dsl.InputPath(str),
    model_out_path: dsl.OutputPath(str),
):
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    X_train = joblib.load(X_train_path)
    y_train = joblib.load(y_train_path)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_out_path)


@dsl.component(base_image="python:3.12-slim", packages_to_install=["scikit-learn", "joblib"])
def evaluate_model(
    X_test_path: dsl.InputPath(str),
    y_test_path: dsl.InputPath(str),
    model_in_path: dsl.InputPath(str),
    metrics_out_path: dsl.OutputPath(str),
) -> float:
    from sklearn.metrics import accuracy_score
    import joblib

    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)
    clf = joblib.load(model_in_path)
    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    with open(metrics_out_path, "w") as f:
        f.write(f"accuracy: {acc:.4f}\n")
    return acc


@dsl.pipeline(name="iris-mlops-pipeline", description="Load, train, and evaluate Iris model")
def iris_pipeline():
    prep = load_and_preprocess_data()
    trainer = train_model(
        X_train_path=prep.outputs["X_train_path"],
        y_train_path=prep.outputs["y_train_path"],
    )
    evaluate_model(
        X_test_path=prep.outputs["X_test_path"],
        y_test_path=prep.outputs["y_test_path"],
        model_in_path=trainer.outputs["model_out_path"],
    )


def register_and_promote_mlflow():
    os.makedirs("ml_pipeline/artifacts", exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///ml_pipeline/mlflow.db")
    mlflow.set_experiment("iris_k8s_experiment")

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    model_name = "IrisClassifier"
    with mlflow.start_run(run_name="rf_iris_run"):
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        acc = float(accuracy_score(y_test, clf.predict(X_test)))

        model_file = "ml_pipeline/artifacts/iris_model.joblib"
        joblib.dump(clf, model_file)

        mlflow.log_param("n_estimators", 50)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact(model_file)
        mlflow.sklearn.log_model(sk_model=clf, name="model", registered_model_name=model_name, skops_trusted_types=["sklearn.tree._tree.Tree"])

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max(int(v.version) for v in versions)

    client.transition_model_version_stage(
        name=model_name,
        version=str(latest_version),
        stage="Staging",
        archive_existing_versions=False,
    )
    client.transition_model_version_stage(
        name=model_name,
        version=str(latest_version),
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Registered {model_name} v{latest_version} and transitioned Staging -> Production (acc={acc:.4f})")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path="ml_pipeline/iris_pipeline.yaml",
    )
    print("Compiled Kubeflow pipeline to ml_pipeline/iris_pipeline.yaml")
    register_and_promote_mlflow()
