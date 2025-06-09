import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from collections import Counter

def train_random_forest_with_children(data, xcol, ycol, hyperparameters_list, experiment_name, regression=True):
    """
    상위 실험 아래에 여러 하이퍼파라미터 조합을 자식 실험으로 기록하는 함수
    
    Args:
        data: 학습할 데이터프레임
        xcol: 특성(X) 컬럼명 리스트 
        ycol: 타겟(y) 컬럼명
        hyperparameters_list: 하이퍼파라미터 딕셔너리의 리스트
        experiment_name: MLflow 실험 이름
        regression: True면 회귀, False면 분류 문제
    
    Returns:
        best_model: 최적의 모델
        child_run_ids: 각 child run의 ID 리스트
    """
    
    # 실험 생성 또는 가져오기
    mlflow.set_experiment(experiment_name)
    
    # 부모 실행 시작
    with mlflow.start_run(run_name="parent_run") as parent_run:
        # 부모 실행의 태그 설정
        mlflow.set_tag("run_type", "parent")
        
        best_model = None
        best_metric = float('-inf') if not regression else float('inf')
        top1_features = []  # 모든 child의 top1 feature를 저장할 리스트
        child_run_ids = []  # child run ID를 저장할 리스트
        
        # 각 하이퍼파라미터 조합에 대해 자식 실행 생성
        for i, hyperparameters in enumerate(hyperparameters_list):
            with mlflow.start_run(run_name=f"child_run_{i}", nested=True) as child_run:
                # child run ID 저장
                child_run_ids.append(child_run.info.run_id)
                
                # 자식 실행의 태그 설정
                mlflow.set_tag("run_type", "child")
                
                # 데이터 준비
                X = data[xcol]
                y = data[ycol]
                
                # 모델 선택 및 학습
                if regression:
                    model = RandomForestRegressor(**hyperparameters)
                else:
                    model = RandomForestClassifier(**hyperparameters)
                    
                model.fit(X, y)
                
                # Feature importance 계산 및 상위 3개 저장
                importances = model.feature_importances_
                feature_importance = dict(zip(xcol, importances))
                top3_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3])
                
                # Top 3 features를 파라미터로 저장
                for rank, (feature, _) in enumerate(top3_features.items(), 1):
                    mlflow.log_param(f"importance_rank_{rank}", feature)
                
                # Top 1 feature를 리스트에 추가
                top1_feature = list(top3_features.keys())[0]
                top1_features.append(top1_feature)
                
                # 예측
                y_pred = model.predict(X)
                
                # 메트릭 계산 및 기록
                if regression:
                    mse = mean_squared_error(y, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y, y_pred)
                    
                    mlflow.log_metric("mse", mse)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("r2", r2)
                    
                    print(f"Child Run {i} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
                    
                    if rmse < best_metric:
                        best_metric = rmse
                        best_model = model
                else:
                    accuracy = accuracy_score(y, y_pred)
                    precision = precision_score(y, y_pred, average='weighted')
                    recall = recall_score(y, y_pred, average='weighted')
                    f1 = f1_score(y, y_pred, average='weighted')
                    
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1", f1)
                    
                    print(f"Child Run {i} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                    
                    if f1 > best_metric:
                        best_metric = f1
                        best_model = model
                
                # 하이퍼파라미터 기록
                for param_name, param_value in hyperparameters.items():
                    mlflow.log_param(param_name, param_value)
                
                # 모델 저장
                mlflow.sklearn.log_model(model, "random_forest_model")
        
        # 부모 실행에 top1 feature들의 빈도수 저장
        top1_counts = dict(Counter(top1_features))
        mlflow.log_param("top1_feature_counts", str(top1_counts))
        
        # 부모 실행에 최적 모델 저장
        if best_model is not None:
            mlflow.sklearn.log_model(best_model, "best_model")
            if regression:
                mlflow.log_metric("best_rmse", best_metric)
            else:
                mlflow.log_metric("best_f1", best_metric)
        
        # child run ID들을 파라미터로 저장
        mlflow.log_param("child_run_ids", str(child_run_ids))
        
        return best_model, child_run_ids

def load_child_model(run_id):
    """
    Child run ID를 사용하여 저장된 모델을 불러오는 함수
    
    Args:
        run_id: Child run의 ID
    
    Returns:
        loaded_model: 불러온 모델
    """
    try:
        model_uri = f"runs:/{run_id}/random_forest_model"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        return loaded_model
    except Exception as e:
        print(f"모델을 불러오는 중 오류 발생: {str(e)}")
        return None

# 사용 예시:
"""
# 하이퍼파라미터 범위 설정
from itertools import product

# 각 하이퍼파라미터의 가능한 값들을 정의
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': list(range(1, 11)),  # 1부터 10까지
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto'],  # 기본값 사용
    'min_samples_leaf': [1]    # 기본값 사용
}

# itertools.product를 사용하여 모든 조합 생성
hyperparameters_list = [
    dict(zip(param_grid.keys(), values))
    for values in product(*param_grid.values())
]

# 실험 실행
best_model, child_run_ids = train_random_forest_with_children(
    data=your_data,
    xcol=your_feature_columns,
    ycol=your_target_column,
    hyperparameters_list=hyperparameters_list,
    experiment_name="RandomForest_Experiment",
    regression=True
)

# 특정 child run의 모델 불러오기
child_model = load_child_model(child_run_ids[0])  # 첫 번째 child run의 모델 불러오기
"""
