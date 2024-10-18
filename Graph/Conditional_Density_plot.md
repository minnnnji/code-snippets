### Conditional_Density_plot 
- 주로 이진 분류일 때 특정변수에서 Yes vs No 분포 차이 볼 때 확인 
- 트리 모델링 후 중요 변수에 대해서 그래프 확인 or Root Node에 대해서 확인 


``` python 
import seaborn as sns

sns.kdeplot(
    data=diamonds,
    x="carat", hue="cut",
    kind="kde", height=6,
    multiple="fill", clip=(0, None),
    palette="ch:rot=-.25,hue=1,light=.75",
)