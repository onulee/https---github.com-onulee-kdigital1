# https://fenderist.tistory.com/168  find 요소 설명
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import requests
from bs4 import BeautifulSoup
import time    # 대기시간 사용을 위해 import
import random  # 랜덤으로 input에 데이터 입력을 위해 import
import pyautogui


# 출력화면이 나타날때까지 대기하는 라이브러리
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
# 브라우저 화면의 상태를 알려주는 라이브러리
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
url = "https://cafe.naver.com/joonggonara?iframe_url=/ArticleSearchList.nhn%3Fsearch.clubid=10050146%26search.searchBy=0%26search.query=%B0%A8%B1%E2"

headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"}
res = requests.get(url,headers=headers)
soup = BeautifulSoup(res.text,"lxml")

print(soup)