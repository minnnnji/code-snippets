from Utils.Config import *
from Pages._Make_Data import make_data

st.set_page_config(
    page_title="데이터 분석 도구",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # 사이드바를 처음부터 접힌 상태로 설정
)

# CSS를 사용하여 사이드바 숨기기
st.markdown("""
    <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# 페이지 선택을 위한 탭 생성
tab1, tab2, tab3 = st.tabs(["Merge Data Frame", "Labeling Data", "Graph Data"])

with tab1:
    st.title("Merge Data Frame")
    make_data()
    
with tab2:
    st.title("Labeling Data")
    st.write("라벨링 데이터 페이지 구현 예정")
    
with tab3:
    st.title("Graph Data")
    st.write("그래프 데이터 페이지 구현 예정")
