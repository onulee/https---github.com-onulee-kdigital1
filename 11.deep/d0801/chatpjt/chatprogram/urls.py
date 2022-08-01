from django.urls import include, path
from . import views

app_name='chatprogram'
urlpatterns = [
    # chat
    path('chat/',views.chat,name='chat'),
    # chat응답json
    path('chat_service/',views.chat_service,name='chat_service'),
]
