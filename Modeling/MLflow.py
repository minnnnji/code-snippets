import mlflow
import os
import mlflow.models
import pandas as pd
import numpy as np
import re

from sklearn.tree import plot_tree, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import *
import statsmodels as sm

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
    
def get_next_run_name(experiment_name, base_name='model'):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    try: 
        max_number = 0
        for run_name in runs['tags.mlflow.runName']:
            match = re.match(rf'{base_name}_(\d+)', run_name)
            if match:
                max_number = max(max_number, int(match.group(1)))
        return f"{base_name}_{max_number + 1}"
    except:
        return f"{base_name}_1"
    
def classification_result(model, y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    precision = precision_score(y, y_pred, average='weighted')

    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", 
                xticklabels=model.classes_, 
                yticklabels=model.classes_)

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    return accuracy, f1, recall, precision, cm

def Decision_Tree_Record(data:pd.DataFrame, xcol:list, ycol:str, experiment_name:str, Regression:bool = True, max_depth = 3, model_name:str = None):
    mlflow.set_tracking_uri("https://ip:port")
    mlflow.set_experiment(experiment_name)
    if model_name == None: model_name = f'{get_next_run_name(experiment_name)}'
    
    with mlflow.start_run(run_name=model_name):
        X = data[xcol]
        y = data[ycol]
        
        if Regression : model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        else: model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        result_dict = {}
        find_node_name(model.tree_, xcol, 0, result_dict)

        # ✅ 결정 트리 시각화
        plt.figure(figsize=(15, 10))
        plot_tree(model, 
                feature_names=xcol, 
                filled=True, 
                fontsize=12, 
                max_depth=3, 
                rounded=True, 
                impurity=False)
        plt.savefig('decision_tree.png')
        plt.close()

        # ✅ 모델 및 파라미터 저장 (MLflow)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params({
            "features": xcol,
            "target": ycol,
            "data_cnt": len(data),
            "n_features": len(X.columns),
            "First_Feature": result_dict.get('1'),
            "Second_Feature": result_dict.get('2'),
            "Third_Feature": result_dict.get('3'),
        })
        mlflow.log_artifact("decision_tree.png")

        # ✅ 이미지 파일 삭제
        os.remove("decision_tree.png")
        
        if Regression:
            result = data[[ycol]].copy()
            result["pred"] = y_pred

            compare_real_pred_graph(result, x=None, y1=ycol, y2='pred', line=False, save=True)
            mlflow.log_artifact("./images/compare_result.png")
            os.remove("./images/compare_result.png")

            rmse = mean_squared_error(y, y_pred, squared=False)
            r2 = r2_score(y, y_pred)

            mlflow.log_param('model_info', "Regression Decision Tree")
            mlflow.log_metrics({
                'RMSE': round(rmse, 3),
                'R2': round(r2, 2)
            })

            print(f"RMSE: {rmse:.4f}")
            print(f"R2: {r2:.4f}")

        else:
            accuracy, f1, recall, precision, cm = classification_result(model, y, y_pred)

            mlflow.log_param('model_info', "Classification Decision Tree")
            mlflow.log_metrics({
                "Accuracy": round(accuracy, 2),
                "F1": round(f1, 2),
                "Recall": round(recall, 2),
                "Precision": round(precision, 2)
            })

            mlflow.log_artifact("confusion_matrix.png")
            os.remove("confusion_matrix.png")

            print(f"xcols: {xcol}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"Precision: {precision:.4f}")
            print("-" * 50)
    return model

def Random_Forest_Record(data:pd.DataFrame, xcol:list, ycol:str, experiment_name:str, Regression:bool=True, model_name:str=None):
    
    mlflow.set_tracking_uri("http://ip:port")
    mlflow.set_experiment(experiment_name)
    
    if model_name == None: model_name = f'{get_next_run_name(experiment_name)}'
    
    with mlflow.start_run(run_name=model_name):
        X = data[xcol]
        y = data[ycol]
        
        if Regression: model = RandomForestRegressor(n_estimators=100, random_state=42)
        else: model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns, 
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=((10, 6)))
        sns.barplot(data=feature_importance[:20], x='importance', y='feature', orient='h')
        plt.title("Top 20 Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()
        
        y_pred = model.predict(X)
        signature = mlflow.models.infer_signature(X, y)
        
        mlflow.log_params({'First_Feature': feature_importance['feature'][0],
                           'Second_Feature': feature_importance['feature'][1],
                           'Third_Feature': feature_importance['feature'][2], 
                           'Top 20 Feature Importance' : feature_importance['feature'][:20].tolist(),
                           'Model Info':model_name}) 
        mlflow.log_artifact("feature_importance.png")
        os.remove('feature_importance.png')
        
        if Regression:
            rmse = mean_squared_error(y, y_pred, squared=False)
            r2 = r2_score(y, y_pred)
            mlflow.log_metrics({'RMSE': round(rmse, 3),
                                'R2': round(r2, 2)} )
            print(f"RMSE: {rmse:.4f}")
            print(f"R2: {r2:.4f}")
        else:
            accuracy = accuracy_score(y, y_pred)
            class_report = classification_report(y, y_pred)
            
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_text(class_report, "classification_report.txt")
            
        mlflow.sklearn.log_model(model, "Model", signature=signature, input_example=X.iloc[:5])

    return model

def OLS_Record(data:pd.DataFrame, xcol:list, ycol:str, experiment_name:str, model_name:str=None):
    mlflow.set_tracking_uri("https://ip:port")
    mlflow.set_experiment(experiment_name)
    if model_name == None: model_name = f'{get_next_run_name(experiment_name)}'
    
    with mlflow.start_run(run_name=model_name):
        X = sm.add_constant(data[xcol])
        y = data[ycol]
        
        model = sm.OLS(y, X).fit()
        
        y_pred = model.predict(X)
        
        # Metiric
        rmse = mean_squared_error(y, y_pred, squared=False)
        r2 = r2_score(y, y_pred)
        mlflow.log_metrics({'RMSE': round(rmse, 3),
                            'R2': round(r2, 2)} )
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

        
        model_summary = model.summary().as_text()
        mlflow.log_text(model_summary, "model_summary.txt")
        
        signature = mlflow.models.infer_signature(X, y)
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X.iloc[:2])
        mlflow.log_param("Model info", model_name)
        
        print("-" * 50)
        print("model_summary: ")
        print(model_summary)
        
    return model

def get_model(experiment_name:str, model_name:str, save_mode=False):
    """ MLflow실험에서 모델을 검색하여 반환하는 함수  

    Args:
        model_name (str): 검색할 모델이름
        experiment_name (str): 검색할 실험 이름
        save_mode (bool, optional): 일반 모델 검색시 0, artifact 저장시 1(함수 내에서만 사용), Defaults to 0.
    """
    try: 
        mlflow.set_tracking_uri("http://ip:port")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            raise Exception(f"실험 '{experiment_name}'을 찾을 수 없습니다.")
        
        runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{model_name}'")
        
        if runs.empty:
            raise Exception(f"실험 '{experiment_name}'에서 모델 '{model_name}'을 찾을 수 없습니다.")
        
        run = runs.iloc[0]
        model_url= f"runs:/{run.run_id}/model"
        if save_mode: model = mlflow.pyfunc.load_model(model_url)
        else: model = mlflow.sklearn.load_model(model_url)
        
        return model
    except Exception as e:
        print(f"모델을 찾을 수 없습니다.\n{str(e)}")
        print("실험 이름과 모델 이름을 확인해주세요.")
        return None
    
def save_artifact_to_mlflow(file_path:str, experiment_name:str, model_name:str):
    """파일을 MLflow 실험의 아티팩트로 저장하는 함수

    Args:
        file_path (str): 파일 경로
        experiment_name (str): 실험 이름
        model_name (str): 모델 이름
    """
    try:
        model = get_model(experiment_name=experiment_name, model_name=model_name, save_mode=True)
        if model is not None:
            with mlflow.start_run(run_id=model.metadata.run_id):
                mlflow.log_artifact(file_path)
                print(f"파일 {file_path}이(가) 실험 '{experiment_name}'/'{model_name}'내에 저장되었습니다.")
                
    except Exception as e:
        print(f"아티팩트 저장 중 오류 발생:{str(e)}")
        
        
def find_node_name(tree, feature_names, node_id, result_dict, count=1, max_nodes=3):
    if node_id == -1 or count > max_nodes:
        return  # 노드 개수가 최대 개수를 초과하면 종료

    feature_index = tree.feature[node_id]

    if feature_index != -2:  # 리프 노드가 아닌 경우
        feature_name = feature_names[feature_index]
        result_dict[str(count)] = feature_name
        count += 1

    # 왼쪽 자식 노드 탐색
    left_child = tree.children_left[node_id]
    if count <= max_nodes and left_child != -1:
        find_node_name(tree, feature_names, left_child, result_dict, count, max_nodes)

    # 오른쪽 자식 노드 탐색
    right_child = tree.children_right[node_id]
    if count <= max_nodes and right_child != -1:
        find_node_name(tree, feature_names, right_child, result_dict, count, max_nodes)

def compare_real_pred_graph(data, x=None, y1=None, y2='pred', figsize=(8, 4), line=False, save=True):
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(GridSpec(1, 1)[0])

    if x is None:
        data = data.reset_index()
        x = 'index'

    sns.scatterplot(data=data, x=x, y=y1, ax=ax1, label='real')
    if line:
        sns.lineplot(data=data, x=x, y=y1, ax=ax1)

    ax2 = ax1.twinx()
    sns.scatterplot(data=data, x=x, y=y2, ax=ax2, color='r', label='pred')
    if line:
        sns.lineplot(data=data, x=x, y=y2, ax=ax2, color='r')

    plt.xticks(rotation=45)
    (plt.savefig("./images/compare_result.png") if save else plt.show())