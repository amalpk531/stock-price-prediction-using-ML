from django.db import models
from django.utils import timezone
from datetime import timedelta
from django.contrib.auth.models import User

# Create your models here.
class Contact(models.Model):
    name=models.CharField(max_length=30,null=False, blank=False)
    email=models.EmailField()
    description=models.TextField()
    def __str__(self) :
        return f'Message from {self.name}'
    
#otp model
class OTP(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    otp = models.CharField(max_length=6)
    
    def is_expired(self):
        now = timezone.now()
        expiry_time = self.created_at + timedelta(minutes=5)
        return now > expiry_time