"""
validate_data.py
Entry-point wrapper — delegates to src.data.clean.
Run:  python validate_data.py
"""
from src.data.clean import validate, load_db
validate()
load_db()
