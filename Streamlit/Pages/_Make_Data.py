from Utils.Config import *
import os

def make_data():
    # 파일 경로 입력
    file_path = st.text_input("데이터 파일 경로를 입력하세요", placeholder="예: C:/Users/data.csv")
    
    if file_path and os.path.exists(file_path):
        # 파일 확장자에 따라 적절한 읽기 함수 선택
        file_extension = file_path.split('.')[-1].lower()
        try:
            if file_extension == 'csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                st.error("지원하지 않는 파일 형식입니다. CSV 또는 Excel 파일을 사용해주세요.")
                return
                
            # 컬럼 선택을 위한 드롭다운
            col1, col2, col3 = st.columns(3)
            
            with col1:
                groupby_col = st.selectbox(
                    "그룹화할 컬럼을 선택하세요",
                    options=df.columns.tolist()
                )
                
            with col2:
                value_col = st.selectbox(
                    "집계할 값 컬럼을 선택하세요",
                    options=df.columns.tolist()
                )
                
            with col3:
                agg_func = st.selectbox(
                    "집계 함수를 선택하세요",
                    options=['sum', 'mean', 'count', 'min', 'max']
                )
                
            # 선택된 옵션으로 데이터 그룹화
            if st.button("데이터 그룹화"):
                grouped_df = df.groupby(groupby_col)[value_col].agg(agg_func).reset_index()
                st.write("그룹화된 데이터:")
                st.dataframe(grouped_df)
                
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {str(e)}")
    elif file_path:
        st.error("입력한 파일 경로가 존재하지 않습니다.")