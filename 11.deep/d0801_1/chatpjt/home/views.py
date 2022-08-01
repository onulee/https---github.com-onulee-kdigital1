from django.shortcuts import render

# 메인페이지 호출
def index(request):
    return render(request,'index.html')
