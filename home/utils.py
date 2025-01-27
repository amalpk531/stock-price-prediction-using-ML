from django.conf import settings

import random
import string
from django.core.mail import send_mail

def generate_otp():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=6))

def send_otp_email(user_data, otp):
    subject = 'Your OTP for Verification'
    message = f'Your OTP is: {otp}. This OTP will expire in 5 minutes.'
    from_email = 'iscalled531@gmail.com'  # Replace with your Gmail
    recipient_list = [user_data['email']]
    
    send_mail(subject, message, from_email, recipient_list)