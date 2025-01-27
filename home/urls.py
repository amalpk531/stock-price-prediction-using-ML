from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('prediction/', views.prediction, name='prediction'),
    path('login/', views.handlelogin, name='login'),
    path('logout/',views.handlelogout,name='handlelogout'),
    path('signup/',views.handlesignup,name='handlesignup'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('verify-signup-otp/', views.verify_signup_otp, name='verify_signup_otp'),
]