from django.urls import include,path
from . import views

app_name='fboard'
urlpatterns = [
    path('<int:nowpage>/fList/',views.fList,name='fList'),
    path('<int:nowpage>/fWrite/',views.fWrite,name='fWrite'),
    path('<int:nowpage>/<str:f_no>/fView',views.fView,name='fView'),
    # 답글쓰기
    path('<int:nowpage>/<str:f_no>/fReply',views.fReply,name='fReply'),
    # 삭제
    path('<int:nowpage>/<str:f_no>/fDelete',views.fDelete,name='fDelete'),
    # 수정
    path('<int:nowpage>/<str:f_no>/fUpdate',views.fUpdate,name='fUpdate'),
    # 공공데이터
    path('public_list/',views.public_list,name='public_list'),
    # chat
    path('chat/',views.chat,name='chat'),
    # chat응답json
    path('chat_service/',views.chat_service,name='chat_service'),
]
