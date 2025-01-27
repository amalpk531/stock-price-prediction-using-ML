import matplotlib
matplotlib.use('Agg')  # Set matplotlib backend to 'Agg' for non-GUI usage
from django.shortcuts import render, HttpResponse
from datetime import datetime, timedelta
import yfinance as yf
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import MinMaxScaler  # Add this import
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.dates as mdates
#--add-b-my-------------------------- 
from django.core.mail import message
from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib import messages
from .models import Contact
# below import is done for sending emails
from django.conf import settings
from django.core.mail import send_mail
from django.core import mail
from django.core.mail.message import EmailMessage
#-add-b-my-------------- 
#----sentiment analysis----------------#
from textblob import TextBlob
import feedparser
from django.http import JsonResponse
#---login check-----------#
from django.contrib.auth.decorators import login_required
from .models import OTP
from .utils import generate_otp, send_otp_email
#
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib import messages
import re
from .models import OTP
from .utils import generate_otp, send_otp_email
from django.db import transaction
from django.utils import timezone


def fetch_and_analyze_sentiment(ticker):
    try:
        # Fetch news from Yahoo Finance RSS feed
        rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        feed = feedparser.parse(rss_url)
        total_score = 0
        num_articles = 0
        
        # Analyze first 5 articles
        for entry in feed.entries[:5]:
            # Combine title and summary for analysis
            text = f"{entry.title} {entry.summary}"
            analysis = TextBlob(text)
            total_score += analysis.sentiment.polarity
            num_articles += 1
        
        if num_articles == 0:
            return {
                "status": "warning",
                "message": "No articles found",
                "sentiment": "Neutral"
            }
        
        # Calculate average sentiment
        avg_score = total_score / num_articles
        
        # Determine sentiment
        if avg_score > 0.1:
            sentiment = "Positive"
        elif avg_score < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return {
            "status": "success",
            "sentiment": sentiment,
            "score": round(avg_score, 2)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

#endsentiment analysis----------------#


# Load the model
model_path = os.path.join('static', 'updated_latest_oneday.h5')
model = tf.keras.models.load_model(model_path)
#----------------------------------------------#
def remove_last_predicted_row():
    """Remove the last row if it was a prediction (has 0 volume)"""
    try:
        df = pd.read_csv('data.csv', skiprows=[1,2])  # Skip ticker and date rows
        if df.iloc[-1]['Volume'] == 0:  # Check if last row was a prediction
            df = df.iloc[:-1]  # Remove last row
            # Save back to CSV with original format
            with open('static\data.csv', 'w') as f:
                df.to_csv(f, index=False, header=False)
    except Exception as e:
        print(f"Error removing last prediction: {e}")

def append_prediction_to_csv(date, close, high, low, open_price):
    """Append predicted values to data.csv"""
    try:
        # First remove any existing prediction
        remove_last_predicted_row()
        
        # Format the new row
        new_row = f"{date},{close},{high},{low},{open_price},0\n"  # Volume set to 0 to mark as prediction
        
        # Append the new row to the file
        with open('static\data.csv', 'a') as f:
            f.write(new_row)
            
    except Exception as e:
        print(f"Error appending prediction: {e}")
#----------------------------------------------#
# Create your views here.
def home(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    if request.method=="POST":
        fname=request.POST.get("name")
        femail=request.POST.get("email")
        desc=request.POST.get("desc")
        query=Contact(name=fname,email=femail,description=desc)
        query.save()
        messages.success(request, "Thanks For Reaching Us! We will get back to you soon....")
        return redirect('/contact')
    return render(request,'contact.html')

from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib import messages
import re
from .models import OTP
from django.db import transaction
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from django.contrib import messages
import re
from .models import OTP
from .utils import generate_otp, send_otp_email
from django.db import transaction
from django.utils import timezone


def handlelogin(request):
    if request.method == "POST":
        uname = request.POST.get("username")
        pass1 = request.POST.get("pass1")
        
        myuser = authenticate(username=uname, password=pass1)
        if myuser is not None:
            if myuser.is_active:
                login(request, myuser)
                messages.success(request, "Login Successful")
                return redirect('/')
            else:
                messages.warning(request, "Account is inactive. Please verify your email.")
                return redirect('/login')
        else:
            messages.warning(request, "Invalid Credentials")
            return render(request, 'login.html')
    return render(request, 'login.html')

def handlesignup(request):
    if request.method == "POST":
        uname = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("pass1")
        confirmpassword = request.POST.get("pass2")

        if password != confirmpassword:
            messages.warning(request, "Passwords do not match")
            return redirect('/signup')

        if not re.search(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@.]{5,}$', password):
            messages.warning(request, "Password must be at least 5 characters long and include both letters and numbers.")
            return redirect('/signup')

        if User.objects.filter(username=uname).exists():
            messages.info(request, "Username is already taken")
            return redirect('/signup')
        if User.objects.filter(email=email).exists():
            messages.info(request, "Email is already registered")
            return redirect('/signup')

        try:
            request.session['temp_user_data'] = {
                'username': uname,
                'email': email,
                'password': password
            }
            otp_code = generate_otp()
            temp_otp = OTP.objects.create(
                otp=otp_code,
                user=None
            )
            request.session['temp_otp_id'] = temp_otp.id
            send_otp_email({'email': email}, otp_code)
            messages.success(request, "Please check your email for OTP verification.")
            return redirect('/verify-signup-otp')
        except Exception as e:
            messages.error(request, f"Error sending OTP: {str(e)}")
            return redirect('/signup')

    return render(request, 'signup.html')


from django.contrib.auth.hashers import make_password
from django.core.exceptions import ValidationError
from django.db import transaction

def verify_signup_otp(request):
    temp_user_data = request.session.get('temp_user_data')
    temp_otp_id = request.session.get('temp_otp_id')

    if not temp_user_data or not temp_otp_id:
        messages.error(request, "Please signup first")
        return redirect('/signup')

    if request.method == "POST":
        entered_otp = request.POST.get('otp').strip()
        print (entered_otp)
       
        otp_obj = OTP.objects.get(id=temp_otp_id)
        print(otp_obj)
        
        try:
            if otp_obj.is_expired():
                messages.warning(request, 'OTP has expired. Please signup again.')
                otp_obj.delete()
                del request.session['temp_user_data']
                del request.session['temp_otp_id']
                return redirect('/signup')

            with transaction.atomic():
                if entered_otp == otp_obj.otp:
                    # Create user with hashed password
                    myuser = User.objects.create(
                        username=temp_user_data['username'],
                        email=temp_user_data['email'],
                        password=make_password(temp_user_data['password'])
                    )
                    myuser.is_active = True
                    myuser.save()
                    
                    otp_obj.delete()
                    del request.session['temp_user_data']
                    del request.session['temp_otp_id']
                
                    messages.success(request, "Account verified successfully! Please login.")
                    return redirect('/login')
                else:
                    raise ValidationError('Invalid OTP. Please try again.')
        except OTP.DoesNotExist:
            messages.error(request, 'Something went wrong. Please signup again.')
        except ValidationError as e:
            messages.error(request, str(e))
        except Exception as e:
            messages.error(request, f"An unexpected error occurred: {str(e)}")
        
        # If we reach here, something went wrong, so we redirect to signup
        return redirect('/signup')

    return render(request, 'verify_signup_otp.html')


def handlelogout(request):
    logout(request)
    messages.info(request,"Logout Successful")
    return redirect('/')


@login_required(login_url='/login')
def dashboard(request):
    return render(request, 'dashboard.html')

def fetch_data(stock_symbol):
    """Fetch historical stock data from Yahoo Finance with MA100."""
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = "2015-01-01"
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    # Calculate 100-day moving average
    data['MA100'] = data['Close'].rolling(window=100).mean()
    data.to_csv('static/data.csv')
    return data[['Open', 'High', 'Low', 'Close', 'MA100']]

def calculate_accuracy_metrics(y_true, y_pred):
    """Calculate various accuracy metrics for the model predictions."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def prepare_model_input(data, time_steps=60):
    """Prepare the last 60 days of data for model prediction with improved scaling."""
    # Extract relevant columns for scaling
    data_to_scale = data[['Open', 'High', 'Low', 'Close']].copy()
    
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(data_to_scale)
    
    # Get the last 60 days
    last_60_days = scaled_data[-time_steps:]
    
    # Reshape for LSTM input [samples, time steps, features]
    model_input = np.array([last_60_days])
    return model_input, scaler

#weighted score calculation-------------------#
def calculate_weighted_prediction(predicted_close, sentiment_score, lstm_weight=0.85, sentiment_weight=0.15):
    """
    Calculate weighted prediction combining LSTM and sentiment analysis.
    
    Args:
        predicted_close (float): LSTM model's predicted closing price
        sentiment_score (float): Sentiment score (-1 to 1)
        lstm_weight (float): Weight given to LSTM prediction (default 0.7)
        sentiment_weight (float): Weight given to sentiment impact (default 0.3)
    
    Returns:
        float: Weighted prediction combining both signals
    """
    # Convert sentiment score to a price impact factor
    # Sentiment score is between -1 and 1, we'll scale it to a percentage impact
    sentiment_impact = 1 + sentiment_score  # This converts range to 0-2
    
    # Calculate the weighted components
    lstm_component = predicted_close * lstm_weight
    sentiment_component = predicted_close * sentiment_impact * sentiment_weight
    
    # Combine the components
    weighted_prediction = lstm_component + sentiment_component
    
    return weighted_prediction

#-----------------------end------------------------------------------------------------------
def create_enhanced_plot(data, predicted_next_day, stock_symbol, tomorrow_str, accuracy_metrics):
    """Create an enhanced plot with trend analysis and metrics."""
    plt.figure(figsize=(18, 10))
    
    # Calculate trend metrics
    current_price = float(data['Close'].iloc[-1])
    ma100_price = float(data['MA100'].iloc[-1])
    trend = "UPTREND" if current_price > ma100_price else "DOWNTREND"
    distance_from_ma = ((current_price - ma100_price) / ma100_price) * 100
    
    # Set colors based on trend
    bg_color = (0, 1, 0, 0.05) if trend == "UPTREND" else (1, 0, 0, 0.05)
    
    # Set background color
    ax = plt.gca()
    ax.set_facecolor(bg_color)
    
    # Calculate y-axis limits for the background
    min_price = min(
        data[['Open', 'High', 'Low', 'Close']].iloc[-30:].min().min(),
        predicted_next_day[0].min()
    ) * 0.99  # Add 1% padding
    
    max_price = max(
        data[['Open', 'High', 'Low', 'Close']].iloc[-30:].max().max(),
        predicted_next_day[0].max()
    ) * 1.01  # Add 1% padding
    
    # Add shaded background
   
    
    # Convert datetime index for proper plotting
    last_30_dates = data.index[-30:]
    predicted_date = datetime.strptime(tomorrow_str, '%Y-%m-%d')

    # Plot actual prices for all OHLC values
    plt.plot(data.index[-30:], data['Open'].iloc[-30:], label='Actual Open', color='blue', alpha=0.7)
    plt.plot(data.index[-30:], data['High'].iloc[-30:], label='Actual High', color='green', alpha=0.7)
    plt.plot(data.index[-30:], data['Low'].iloc[-30:], label='Actual Low', color='red', alpha=0.7)
    plt.plot(data.index[-30:], data['Close'].iloc[-30:], label='Actual Close', color='purple', alpha=0.7)
    
    # Plot MA100
    plt.plot(data.index[-30:], data['MA100'].iloc[-30:], 
             label='100-day Moving Average', 
             color='orange', 
             linestyle='--', 
             linewidth=2)
    
    # Get last actual points for connecting lines
    last_actual_open = float(data['Open'].iloc[-1])
    last_actual_high = float(data['High'].iloc[-1])
    last_actual_low = float(data['Low'].iloc[-1])
    last_actual_close = float(data['Close'].iloc[-1])
    
    # Plot predicted points and connection lines
    # Open price
    plt.scatter(predicted_date, predicted_next_day[0, 0], color='blue', s=100, label='Predicted Open')
    plt.plot([data.index[-1], predicted_date], 
            [last_actual_open, predicted_next_day[0, 0]], 
            color='blue', linestyle='--', alpha=0.5)
    
    # High price
    plt.scatter(predicted_date, predicted_next_day[0, 1], color='green', s=100, label='Predicted High')
    plt.plot([data.index[-1], predicted_date], 
            [last_actual_high, predicted_next_day[0, 1]], 
            color='green', linestyle='--', alpha=0.5)
    
    # Low price
    plt.scatter(predicted_date, predicted_next_day[0, 2], color='red', s=100, label='Predicted Low')
    plt.plot([data.index[-1], predicted_date], 
            [last_actual_low, predicted_next_day[0, 2]], 
            color='red', linestyle='--', alpha=0.5)
    
    # Close price
    plt.scatter(predicted_date, predicted_next_day[0, 3], color='purple', s=100, label='Predicted Close')
    plt.plot([data.index[-1], predicted_date], 
            [last_actual_close, predicted_next_day[0, 3]], 
            color='purple', linestyle='--', alpha=0.5)
    
    # Add annotations for predicted values
    for i, (value, color) in enumerate(zip(predicted_next_day[0], ['blue', 'green', 'red', 'purple'])):
        label = ['Open', 'High', 'Low', 'Close'][i]
        plt.annotate(f'{label}: {value:.2f}', 
                    xy=(predicted_date, value),
                    xytext=(10, 10), 
                    textcoords='offset points',
                    color=color,
                    bbox=dict(facecolor='white', edgecolor=color, alpha=0.7))
    
    # Configure axes and formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    # Add title and labels
    plt.title(f'{stock_symbol} Stock Price Prediction with Trend Analysis\n'
              f'Current Trend: {trend} ({distance_from_ma:.2f}% from MA100)\n'
              f'Next Day: {tomorrow_str}', 
              pad=20)
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    # Add metrics text boxes
    trend_metrics = (
        f'Trend Analysis:\n'
        f'Current Price: {current_price:.2f}\n'
        f'MA100: {ma100_price:.2f}\n'
        f'Distance from MA100: {distance_from_ma:.2f}%'
    )
    plt.text(0.02, 0.98, trend_metrics, 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    metrics_text = (
        f'Model Metrics:\n'
        f'RMSE: {accuracy_metrics[1]:.2f}\n'
        f'MAE: {accuracy_metrics[2]:.2f}\n'
        f'R²: {accuracy_metrics[3]:.2f}'
    )
    plt.text(0.02, 0.8, metrics_text, 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    # Save plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    buffer.close()
    
    return graph_base64, trend, distance_from_ma
@login_required(login_url='/login')
def prediction(request):
    # Add these date calculations at the start of the function
    today = datetime.today()
    tomorrow = today + timedelta(days=1)
    tomorrow_str = tomorrow.strftime('%Y-%m-%d')

    if request.method == "POST":
        stock_symbol = request.POST.get('stock_symbol')
        
        try:
            # Fetch sentiment analysis (keeping your existing code)
            sentiment_result = fetch_and_analyze_sentiment(stock_symbol)
            
            # Fetch data
            data = fetch_data(stock_symbol)
            
            # Prepare input and make prediction
            model_input, scaler = prepare_model_input(data)
            scaled_prediction = model.predict(model_input)
            
            # Inverse transform the prediction
            prediction = scaler.inverse_transform(scaled_prediction)
            predicted_next_day = prediction[0]
            
            # Calculate accuracy metrics using the last 60 days of data
            y_true = data[['Open', 'High', 'Low', 'Close']].iloc[-60:].values
            y_pred = model.predict(model_input)
            y_pred = scaler.inverse_transform(y_pred)
            accuracy_metrics = calculate_accuracy_metrics(y_true[-1:], y_pred)
            
            # Create enhanced plot
            graph_base64, trend, distance_from_ma = create_enhanced_plot(
                data, prediction, stock_symbol, tomorrow_str, accuracy_metrics
            )
            
            #--------------------weighted score calculation-------------------#
            weighted_prediction = calculate_weighted_prediction(
            predicted_close=predicted_next_day[3],  # Close price
            sentiment_score=sentiment_result.get("score", 0),  # Sentiment score
            lstm_weight=0.85,  # 85% weight to LSTM prediction
            sentiment_weight=0.15 # 15% weight to sentiment
)
#----------------------end-----------------------------------------#
  # Append prediction to CSV (keeping your existing code)
            append_prediction_to_csv(
                tomorrow_str,
                weighted_prediction,  # Close
                predicted_next_day[1],  # High
                predicted_next_day[2],  # Low
                predicted_next_day[0]   # Open
            )
            # Prepare context
            context = {
                "predicted_open": round(predicted_next_day[0], 2),
                "predicted_high": round(predicted_next_day[1], 2),
                "predicted_low": round(predicted_next_day[2], 2),
                "predicted_close": round(predicted_next_day[3], 2),
                "graph": graph_base64,
                "symbol": stock_symbol,
                "t_date": tomorrow_str,
                "trend": trend,
                "distance_from_ma": round(distance_from_ma, 2),
                "rmse": round(accuracy_metrics[1], 2),
                "mae": round(accuracy_metrics[2], 2),
                "r2": round(accuracy_metrics[3], 2),
                "sentiment": sentiment_result["sentiment"],
                "sentiment_score": sentiment_result.get("score", 0),
                "weighted_prediction":(weighted_prediction)
            }
            
            return render(request, 'prediction.html', context)
            
        except Exception as e:
            error_message = f"Error processing {stock_symbol}: {str(e)}"
            return render(request, 'prediction.html', {"error": error_message})
    
    
    return render(request, 'prediction.html')