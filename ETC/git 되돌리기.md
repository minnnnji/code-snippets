### STEP 1. 
    $ git log -5 --pretty=format:"%h - %an, %ar : %s"
    이렇게하면 최근 5개의 리스트가 뜨는데 그중에 돌아가고 싶은 시점으로 가면된다.

### STEP 2. 
만약에  
    
    aaa -~~ 5 minutes ago : ~~~
    bbb -~~ 15 minutes ago : ~~~
    ccc -~~ 35 minutes ago : ~~~

이렇게 있을때 5분전에 올린 **aaa** 만 지우고 싶다면

`$ git reset bbb`

### STEP 3. 

`$ git reset --hard`

`$ git reset --soft` 를 하면 기록이 남는다. 
작업보존을 하려면 이렇게 하고 이걸 이용해 다시 올릴 수 있다.

 
### STEP 3 - not necessarily. 
`& git clean -fd`
디렉토리 포함하여 삭제해준다.


    -f, --force : 강제로 삭제
    -d : untracked directory 삭제

### STEP 4 - not necessarily. 
`$ git push -f`
    
    local repo의 브랜치를 remote repo에 업로드 (-f :강제로)