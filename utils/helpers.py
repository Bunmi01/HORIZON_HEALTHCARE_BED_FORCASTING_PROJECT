from datetime import datetime
import pytz

def get_current_datetime():
    """Get current datetime in UTC"""
    timezone = pytz.timezone('UTC')
    return datetime.now(timezone)

def format_accuracy_class(mape):
    """Get CSS class for accuracy"""
    if mape <= 10:
        return "Good accuracy"
    if mape <= 20:
        return ("Fair Accuracy")
    else:
        return ("Poor accuracy")
    

def format_accuracy_text(mape):
    """Get text for accuracy"""
    if mape <= 10:
        return "Excellent"
    if mape <= 20:
        return ("Good")
    else:
        return ("Fair")