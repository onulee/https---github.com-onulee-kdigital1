import urllib.request as req
import gzip, os, os.path
import struct

# csv생성함수
def to_csv(name,maxdata):
    # 파일 읽어오기 - 6만개 데이터를 가지고 있음.
    img_f = open("./10.mlearn/mnist/"+name+"-images-idx3-ubyte","rb")
    lbl_f = open("./10.mlearn/mnist/"+name+"-labels-idx1-ubyte","rb")
    # csv파일로 저장
    csv_f = open("./10.mlearn/mnist/"+name+".csv","w",encoding="utf-8")

    # 헤더 정보읽기 - 정수로 4byte씩 2개를 읽어옴.
    mag,lbl_count = struct.unpack(">II",lbl_f.read(8))
    mag,img_count = struct.unpack(">II",img_f.read(8))
    rows,cols = struct.unpack(">II",img_f.read(8))
    pixels = rows * cols  # 28*28=784

    res=[]
    # 이미지 데이터를 읽고, csv파일에 저장
    for idx in range(lbl_count):  # 6만개 데이터
        if idx > maxdata: break    # 1-6만개 데이터  csv 5000개만 저장
        label = struct.unpack("B",lbl_f.read(1))[0]
        # 이미지의 데이터 값
        bdata = img_f.read(pixels)
        sdata = list(map(lambda a: str(a),bdata))
        csv_f.write(str(label)+",") # csv 0,0,0,0,0,1,124,255,120,.....
        csv_f.write(",".join(sdata)+"\r\n")

    csv_f.close()    
    lbl_f.close()
    img_f.close()
    print(name+": csv파일생성완료")

# 함수 호출
to_csv("train",1000)
to_csv("t10k",500)

