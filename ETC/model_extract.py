import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import mlflow
import mlflow.sklearn
import json
import streamlit as st
import yaml

class ModelInfo:
    def __init__(self, model, feature_columns, target_column, model_type=None, model_params=None):
        """
        모델과 관련 정보를 저장하는 클래스
        
        Args:
            model: 학습된 모델
            feature_columns: 특성(X) 컬럼명 리스트
            target_column: 타겟(y) 컬럼명
            model_type: 모델 종류 (예: 'DecisionTreeClassifier', 'DecisionTreeRegressor', 'OLS')
            model_params: 모델 파라미터 (선택사항)
        """
        self.model = model
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.model_type = model_type
        self.model_params = model_params

    def save_to_yaml(self, filepath):
        """
        모델 정보와 모델을 저장
        
        Args:
            filepath: 저장할 파일 경로 (YAML과 joblib 파일이 같은 경로에 저장됨)
        """
        
        
        # YAML 정보 저장
        info_dict = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_type': self.model_type,
            'model_params': self.model_params
        }
        
        yaml_path = filepath
        model_path = filepath.replace('.yaml', '.joblib')
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(info_dict, f, allow_unicode=True, default_flow_style=False)
            
        # 모델 저장
        joblib.dump(self.model, model_path)

def create_model_info(model) -> ModelInfo:
    """
    DecisionTreeClassifier/Regressor나 sm.OLS 모델에서 ModelInfo 객체를 생성합니다.
    
    Args:
        model: DecisionTreeClassifier, DecisionTreeRegressor 또는 sm.OLS 모델 인스턴스
    Returns:
        ModelInfo: 모델 정보가 담긴 객체
    """
    from sklearn.tree import DecisionTreeRegressor
    
    model_type = model.__class__.__name__
    if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        # DecisionTree 모델인 경우
        feature_columns = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else None
        
        if isinstance(model, DecisionTreeClassifier):
            # 분류 모델인 경우 클래스 정보 저장
            target_column = model.classes_.tolist() if hasattr(model, 'classes_') else None
        else:
            # 회귀 모델인 경우
            target_column = 'regression_target'
            
        model_params = model.get_params()
        
    else:
        # sm.OLS인 경우 
        feature_columns = model.model.exog_names if hasattr(model.model, 'exog_names') else None
        target_column = model.model.endog_names if hasattr(model.model, 'endog_names') else None
        model_params = {
            'nobs': float(model.nobs),
            'rsquared': float(model.rsquared),
            'rsquared_adj': float(model.rsquared_adj)
        }

    model_info = ModelInfo(model, feature_columns, target_column, model_type, model_params)
    
    # JSON 파일로 저장
    model_info.save_to_json('model_info.json')
    
    return model_info

def streamlit_model_prediction():
    st.title('저장된 모델로 예측하기')
    
    # 모델 경로 입력
    save_dir = st.text_input('저장된 모델 경로:', 'model_artifacts')
    
    # 예측용 데이터 업로드
    uploaded_file = st.file_uploader("예측할 데이터를 업로드하세요 (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        st.write("데이터 미리보기:")
        st.dataframe(new_data.head())
        
        if st.button('예측 시작'):
            try:
                predictions, model_info = load_and_predict(save_dir, new_data)
                
                st.write("예측에 사용된 특성:", model_info.feature_columns)
                st.write("타겟 변수:", model_info.target_column)
                
                # 예측 결과를 데이터프레임에 추가
                results_df = new_data.copy()
                results_df['예측 결과'] = predictions
                
                st.write("예측 결과:")
                st.dataframe(results_df)
                
                # 예측 결과 다운로드 버튼
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="예측 결과 다운로드",
                    data=csv,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f'오류 발생: {str(e)}')

def main():
    streamlit_model_prediction()

if __name__ == '__main__':
    main()
