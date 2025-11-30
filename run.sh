# This command tells Uvicorn to look in the 'app' module for 'main'
# and use the 'app' object inside 'main.py'.
uvicorn app.main:app --reload --app-dir .