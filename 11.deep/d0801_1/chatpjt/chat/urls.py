from django.urls import include,path
from . import views

app_name='chat'
urlpatterns = [
    path('chat/',views.chat,name='chat'),
    # json통신
    path('chat_service/',views.chat_service,name='chat_service'),
]
