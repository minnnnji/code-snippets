### PI System에서 접근 및 데이터 추출 함수 

```python 
import os
import pandas as pd
import datetime
import PIconnect as PI
from tqdm import tqdm
import PIconnect.PIConsts import SummaryType, CalculationBasis
import requests

server = PI.PIServer(server='서버 IP 주소')

def extract_tag(tag, start_time, end_time, d_type='mean', interval = ':60s'):
    d_types = {'mean' : SummaryType.AVERAGE, 
                'max': SummaryType.MAXIMUM, 
                'min':SummaryType.MINIMUM }
    points = server.search(tag)[0]
    data = points.summaries(start_time=start_time, 
                            end_time=end_time, 
                            interval=interval, 
                            summary_type=d_types[d_type] 
                            calculation_basis=CalculationBasis.TIME_WEIGHTED, 
                            time_type=1 )

    data = pd.DataFrame(data).reset_index()
    data.columns = ['TIME', 'VALUE']
    data['TIME'] = (data['TIME'] + datetime.timedelta(hours=9)).values() # 시간 보정
    return data

# example 
current_time = datetime.datetime.now() # or datetime.datetime(2024, 10, 04, 10) -> 24년 10월 4일 10시 
start_time = (current_time - datetime.timedelta(days=1))\ # hours, weeks ... 
            .strftime('%Y-%m-%d %H:%M') 
end_time = current_time.strftime('%Y-%m-%d %H:%M')

extract_tag(tag, start_time, end_time)
```