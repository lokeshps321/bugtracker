import smtplib
from email.mime.text import MIMEText
import aiohttp
import os
from sqlalchemy.orm import Session
from app import models
from app.config import Config
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Load configuration
Config.validate_config()

SMTP_SERVER = Config.SMTP_SERVER
SMTP_PORT = Config.SMTP_PORT
SMTP_USER = Config.SMTP_USER
SMTP_PASSWORD = Config.SMTP_PASSWORD
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK", "")


# --------------------------------------------------------------------------
# FIX: Synchronous function required by app/main.py's BackgroundTasks
# --------------------------------------------------------------------------
def send_email_notification_sync(recipient_email: str, subject: str, body: str):
    """
    Synchronously sends an email notification. Designed to be run 
    in a background task via FastAPI's BackgroundTasks.
    """
    print(f"Attempting to send email to {recipient_email} with subject: {subject}")
    
    if not all([SMTP_SERVER, SMTP_USER, SMTP_PASSWORD]):
        error_msg = f"SMTP credentials missing. Required: SMTP_SERVER, SMTP_USER, SMTP_PASSWORD. Current values - Server: {SMTP_SERVER}, User: {SMTP_USER}"
        print(error_msg)
        logger.error(error_msg)
        return

    try:
        # Create message
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = recipient_email

        print(f"Connecting to SMTP server: {SMTP_SERVER}:{SMTP_PORT}")
        
        # Connect to SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
        print("SMTP connection established")
        
        # Enable TLS
        print("Starting TLS...")
        server.starttls()
        print("TLS started")
        
        # Login
        print(f"Logging in as {SMTP_USER}")
        server.login(SMTP_USER, SMTP_PASSWORD)
        print("Login successful")
        
        # Send email
        print("Sending email...")
        server.send_message(msg)
        print("Email sent successfully")
        
        # Close connection
        server.quit()
        logger.info(f"Email notification sent to {recipient_email} (Subject: {subject})")
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"SMTP Authentication Error: {str(e)}. Please check your email and password."
        print(error_msg)
        logger.error(error_msg)
    except smtplib.SMTPException as e:
        error_msg = f"SMTP Error: {str(e)}"
        print(error_msg)
        logger.error(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error sending email to {recipient_email}: {str(e)}"
        print(error_msg)
        logger.error(error_msg)


# --------------------------------------------------------------------------
# Original async notification function (kept for reference/other endpoints)
# --------------------------------------------------------------------------
async def send_notification(message: str, team: str, sender_email: str, db: Optional[Session] = None):
    # ----------------------
    # Store notification in DB
    # ----------------------
    if db:
        try:
            # Use asyncio.to_thread to avoid blocking event loop for DB operation
            def db_commit():
                db_notification = models.Notification(
                    message=message,
                    recipient_team=team,
                    sender_email=sender_email
                )
                db.add(db_notification)
                db.commit()
            await asyncio.to_thread(db_commit)
            logger.info("Notification stored in database")
        except Exception as e:
            logger.error(f"Failed to store notification: {str(e)}")

    # ----------------------
    # Email notification (uses asyncio.to_thread for safety)
    # ----------------------
    try:
        if all([SMTP_SERVER, SMTP_USER, SMTP_PASSWORD]):
            msg = MIMEText(message)
            msg["Subject"] = "BugFlow Notification"
            msg["From"] = SMTP_USER
            msg["To"] = f"{team}@example.com" # Guess recipient based on team
            
            def send_email():
                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                    server.starttls()
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.send_message(msg)

            await asyncio.to_thread(send_email)
            logger.info(f"Async Email sent to {team}@example.com")
        else:
             logger.warning("SMTP credentials missing. Skipping async email.")

    except Exception as e:
        logger.error(f"Async Email notification failed: {str(e)}")

    # ----------------------
    # Slack notification
    # ----------------------
    if SLACK_WEBHOOK:
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"text": message}
                async with session.post(SLACK_WEBHOOK, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Slack notification failed: {response.status}")
                    else:
                        logger.info("Slack notification sent successfully")
        except Exception as e:
            logger.error(f"Slack notification failed: {str(e)}")
