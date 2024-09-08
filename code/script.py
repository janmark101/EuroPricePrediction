from twilio.rest import Client
from dotenv import load_dotenv
import os


def send_sms(pred_price:int):
    load_dotenv()

    account_sid = os.getenv('SID')
    auth_token = os.getenv('AUTH_TOKEN')

    client = Client(account_sid, auth_token)
    message = client.messages.create(
    messaging_service_sid=os.getenv('SERVICE_SID'),
    body=f"Tomorrow's predicted price for EURO {pred_price:.4f} 🛒💸",
    to=os.getenv('NUMBER'),
    )
    print(message.sid)