### pandas로 안 열리는 엑셀 파일 열기 

| pandas로 안 열리는 엑셀파일 열어서 데이터 프레임으로 저장 
```python

import pythoncom
import win32com.client
import pandas as pd
import numpy as np
import os

def excel_initial(on=False):
    pythoncom.CoInitialize()
    excel = win32com.client.Dispatch('Excel.Application')
    excel.EnableEvents = False
    excel.DisplayAlerts = False
    excel.Visible = on
    return excel

def excel_kill():
    try: os.system("taskkill /f /im EXCEL.exe"); time.sleep(3)
    except:pass
    
def excel_to_DF(file_path, col_idx=0, start_idx=1, sheet=1):
    excel = excel_initial()
    wb = excel.Workbooks.Open(os.path.abspath(file_path), ReadOnly=True)
    ws = wb.Worksheets(sheet)
    values = ws.UsedRange.Rows.values()
    df = pd.DataFrame(values[start_idx:], columns=values[col_idx])
    wb.Close()
    return df

```