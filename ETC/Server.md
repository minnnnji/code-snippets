# 서버에서 통신 
### 01. 서버로 데이터 보낼 때
```python
headers =  { 'User-Agent' : ('Mozilla/5.0 (Windows NT 10.0;Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36')}

file_list = ['파일 이름'] # ex ) './data/example.csv'
files_to_upload = [("files", (file_name, open(os.path.abspath(file_name), "rb").read())) for file_name in File_list]
res = requests.post("http://서버 주소 및 포트/upload-data)", files=files_to_upload, headers=headers)
```

--- 
### 02. 서버에서 데이터 받을 때 

```python 
from fastapi import FastAPI, File, UploadFile
import shutil
import os

app = FastAPI(docs_url='/api/docs', openapi_url='/api/openapi.json')

# 파일을 저장할 디렉터리 경로
UPLOAD_DIR = os.path.join(os.getcwd(), 'uploaded') 
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_unique_filename(directory: str, filename: str) -> str:
    name, extension = os.path.splitext(filename)
    counter = 1

    # 파일이 이미 존재하면, 숫자를 증가시키며 고유한 파일 이름을 찾음
    while os.path.exists(os.path.join(directory, filename)):
        filename = f"{name}_{counter}{extension}"
        counter += 1

    return filename

@app.post("/upload-data")
async def upload_file(file: UploadFile = File(...)):
    saved_files = []
    
    for file in files:
        # 고유한 파일 이름 생성 (UUID 사용)
        unique_filename = get_unique_filename(UPLOAD_DIR, file.filename)
        
        # 파일 저장 경로 설정
        file_location = os.path.join(UPLOAD_DIR, unique_filename)

        # 파일 저장
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        saved_files.append({"filename": unique_filename, "location": file_location})
    
    return {"files": saved_files}
```
