import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pytesseract
from PIL import Image
import pandas as pd
import os
import re
import logging
from tkinterdnd2 import DND_FILES, TkinterDnD
from datetime import datetime

# Setup logging
logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting application...")

# Set Tesseract path
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if not os.path.exists(tesseract_path):
    logging.error(f"Tesseract not found at {tesseract_path}")
    messagebox.showerror("Error", f"Tesseract OCR not found at {tesseract_path}\n\nPlease install Tesseract OCR from:\nhttps://github.com/UB-Mannheim/tesseract/wiki")
else:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    logging.info(f"Tesseract path set to: {tesseract_path}")

# File paths
BANKROLL_FILE = "bankroll.txt"
HISTORY_FILE = "betting_history.csv"

# Load/Save Functions
def load_bankroll():
    if os.path.exists(BANKROLL_FILE):
        with open(BANKROLL_FILE, "r") as f:
            return float(f.read().strip())
    return 200.0

def save_bankroll(br):
    with open(BANKROLL_FILE, "w") as f:
        f.write(str(br))

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["Match", "Odds", "Stake", "Result", "Payout", "Date"])

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)

# Functions
def on_drop(event):
    file_path = event.data
    if file_path.startswith("{") and file_path.endswith("}"):
        file_path = file_path[1:-1]
    if os.path.isfile(file_path) and file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        process_file(file_path)
    else:
        feedback_label.config(text="Invalid file dropped.")
        logging.warning(f"Invalid file dropped: {file_path}")

def adjust_bankroll(amount):
    global bankroll
    bankroll += amount
    if bankroll < 0:
        bankroll = 0
    save_bankroll(bankroll)
    br_label.config(text=f"Bankroll: €{bankroll:.2f}")
    feedback_label.config(text=f"Bankroll adjusted by €{amount:+.2f}")
    logging.info(f"Bankroll adjusted to €{bankroll:.2f}")

def extract_bet_info(image_path):
    try:
        img = Image.open(image_path)
        # Enhanced Tesseract configuration
        custom_config = r'--oem 3 --psm 6 -l eng+est'  # Added Estonian language support
        text = pytesseract.image_to_string(img, config=custom_config)
        logging.info(f"Extracted text from image: {text}")
        
        # Show first few lines of extracted text in feedback
        preview = "\n".join(text.split("\n")[:5])
        feedback_label.config(text=f"Extracted text preview:\n{preview}")
        
        lines = text.split("\n")
        bets = []
        current_bet = {}
        
        # Keywords to skip
        skip_words = ["cash out", "win", "loss", "tulemus", "üksikpanus", "panusesumma", 
                     "võidusumma", "normaalaeg", "normaalaeg tulemus", "eduga võit", 
                     "kaotus", "võit", "võidusumma", "väljamakse"]
        
        # Pattern for stake amount (handles both comma and dot decimals)
        stake_pattern = re.compile(r'(\d+[,\.]\d+)\s*€')
        # Pattern for odds (handles various formats)
        odds_pattern = re.compile(r'([^\d]+?)\s+(\d+[,\.]\d+)\s*$')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            logging.info(f"Processing line: {line}")
            
            # Look for stake amounts
            if "€" in line:
                stake_match = stake_pattern.search(line)
                if stake_match:
                    stake_str = stake_match.group(1).replace(",", ".")
                    stake = float(stake_str)
                    current_bet["stake"] = stake
                    logging.info(f"Found stake: {stake}")
            
            # Look for team names and odds
            odds_match = odds_pattern.search(line)
            if odds_match:
                match = odds_match.group(1).strip()
                odds = float(odds_match.group(2).replace(",", "."))
                
                # Skip if it contains any of the skip words
                if any(word.lower() in match.lower() for word in skip_words):
                    continue
                
                # Check if it's a valid team/player name (at least 3 characters, no special patterns)
                if len(match) >= 3 and not any(char.isdigit() for char in match):
                    current_bet["match"] = match
                    current_bet["odds"] = odds
                    logging.info(f"Found match and odds: {match} - {odds}")
                    
                    # Look ahead for stake in next few lines if not found
                    if "stake" not in current_bet:
                        for next_line in lines[i+1:i+4]:
                            if "€" in next_line:
                                stake_match = stake_pattern.search(next_line)
                                if stake_match:
                                    stake_str = stake_match.group(1).replace(",", ".")
                                    current_bet["stake"] = float(stake_str)
                                    break
                    
                    # If we have both stake and odds, add the bet
                    if "stake" in current_bet:
                        bets.append(dict(current_bet))
                        current_bet = {}
        
        if bets:
            logging.info(f"Successfully extracted {len(bets)} bets: {bets}")
            feedback_text = "Found bets:\n"
            for b in bets:
                feedback_text += f"• {b['match']}\n  Odds: {b['odds']}, Stake: €{b['stake']:.2f}\n"
            feedback_label.config(text=feedback_text)
            return bets
        else:
            feedback_label.config(text=f"No bets found. Text extracted:\n{preview}")
            logging.warning("No bets found in screenshot. Extracted text: " + text)
            return None
        
    except Exception as e:
        feedback_label.config(text=f"Error processing image: {str(e)}\nExtracted text preview:\n{text[:200]}")
        logging.error(f"Error in extract_bet_info: {e}")
        return None 