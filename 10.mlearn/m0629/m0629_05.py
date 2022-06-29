import urllib.request as req
import gzip, os, os.path
# 저장파일 위치
savepath = './10.mlearn/mnist'
baseurl = 'http://yann.lecun.com/exdb/mnist'

# 가져올 파일리스트
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

# 다운로드
if not os.path.exists(savepath): os.mkdir(savepath)

# url에서 파일 다운로드 받아 파일저장
for f in files:
    # 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url = baseurl+"/"+f
    # ./10.mlearn/mnist/train-images-idx3-ubyte.gz
    loc = savepath+"/"+f
    if not os.path.exists(loc):
        # 파일 다운로드 저장시킴 : urlretrieve(가져올url, 파일저장위치)-파일저장함수
        req.urlretrieve(url,loc)
        

# gzip 압축 해제
for f in files:
    # ./10.mlearn/mnist/train-images-idx3-ubyte.gz
    gz_file = savepath + "/"+ f                     # 파일읽기  
    # ./10.mlearn/mnist/train-images-idx3-ubyte 
    raw_file = savepath + "/" + f.replace(".gz","") # 파일 저장
    # 파일 읽어오기
    with gzip.open(gz_file,"rb") as fp:
        body = fp.read()
        # 파일 저장하기
        with open(raw_file,"wb") as w:
            w.write(body)

print("저장완료")            



