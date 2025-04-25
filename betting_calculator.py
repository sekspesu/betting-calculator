import tkinter as tk
from tkinter import filedialog, messagebox, ttk
# import pytesseract # Remove pytesseract
import easyocr # Add easyocr
from PIL import Image
import pandas as pd
import os
import re
import logging
from tkinterdnd2 import DND_FILES, TkinterDnD
from datetime import datetime, timedelta
from team_data import get_team_league, get_all_teams, get_league_teams
from difflib import SequenceMatcher
import numpy as np # <-- Import numpy
import os # <-- Import os needed for icon path

# Setup logging
logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting application...")

# Initialize EasyOCR Reader (English language)
# Doing this globally might be slightly faster if the app is used frequently,
# but requires loading models on startup.
# Consider moving inside extract_bet_info if startup time is critical.
try:
    print("Initializing EasyOCR Reader (this may take a moment)...")
    reader = easyocr.Reader(['en'], gpu=False) # Use gpu=True if you have a compatible GPU and PyTorch installed
    print("EasyOCR Reader initialized.")
except Exception as e:
    logging.error(f"Failed to initialize EasyOCR Reader: {e}")
    messagebox.showerror("EasyOCR Error", f"Failed to initialize EasyOCR Reader: {e}\nPlease ensure EasyOCR and its dependencies (like PyTorch) are correctly installed.")
    reader = None # Set reader to None if initialization fails

# --- Define Global UI Colors ---
# Moved these definitions earlier to ensure they are available for functions
CARD_BG_COLOR = "#3C3C3C"
TEXT_COLOR = "#EAEAEA"
TEXT_MUTED_COLOR = "#A0A0A0"
WIN_COLOR = "#34C759" # <-- Changed back to Green
LOSS_COLOR = "#FF3B30"
ACCENT_COLOR = "#007AFF"
BG_COLOR = "#2D2D2D" # Added main background for consistency if needed elsewhere
# --- End Global UI Colors ---

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
        df = pd.read_csv(HISTORY_FILE)
        # Ensure all required columns exist
        required_columns = ["Match", "League", "Odds", "Stake", "Result", "Payout", "Date"]
        for col in required_columns:
            if col not in df.columns:
                df[col] = "" if col == "League" else 0.0
        return df
    return pd.DataFrame(columns=["Match", "League", "Odds", "Stake", "Result", "Payout", "Date"])

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)

def update_league_filter():
    """Updates the league filter combobox with unique leagues from history."""
    try:
        if not history.empty and 'League' in history.columns:
            # Get unique leagues, convert to string, handle NaN/None, sort
            # Ensure we only add valid league names (not empty strings or 'nan')
            unique_leagues = sorted([str(league) for league in history['League'].unique() if pd.notna(league) and str(league).strip() and str(league).lower() != 'nan'])
            all_leagues = ["All Leagues"] + unique_leagues
            league_combo['values'] = all_leagues
            # Set back to 'All Leagues' if the current value is no longer valid or not present
            current_selection = league_var.get()
            if current_selection not in all_leagues:
                 league_var.set("All Leagues")
            logging.info(f"League filter updated with {len(unique_leagues)} unique leagues.")
        else:
            # Default if history is empty or no 'League' column
            league_combo['values'] = ["All Leagues"]
            league_var.set("All Leagues")
            logging.info("League filter set to default 'All Leagues'. History empty or 'League' column missing.")
    except Exception as e:
        logging.error(f"Error updating league filter: {e}")
        league_combo['values'] = ["All Leagues"] # Fallback
        league_var.set("All Leagues")

def manual_bankroll_change():
    """Opens a dialog to manually set the bankroll amount."""
    dialog = tk.Toplevel(root)
    dialog.title("Set Bankroll")
    dialog.geometry("300x150")
    dialog.configure(bg="#313131") # Match dark theme
    dialog.resizable(False, False)

    # Make dialog modal (prevents interaction with main window)
    dialog.transient(root) 
    dialog.grab_set()

    # Center the dialog
    root.update_idletasks() # Ensure root window size is calculated
    dialog.update_idletasks() # Ensure dialog size is calculated
    root_x = root.winfo_x()
    root_y = root.winfo_y()
    root_width = root.winfo_width()
    root_height = root.winfo_height()
    dialog_width = dialog.winfo_width()
    dialog_height = dialog.winfo_height()
    
    x = root_x + (root_width // 2) - (dialog_width // 2)
    y = root_y + (root_height // 2) - (dialog_height // 2)
    dialog.geometry(f'+{x}+{y}')

    # Widgets inside the dialog
    ttk.Label(dialog, text="Enter new bankroll amount:", 
              background="#313131", foreground="#E0E0E0").pack(pady=(15, 5))

    amount_var = tk.StringVar()
    amount_entry = ttk.Entry(dialog, textvariable=amount_var, width=20,
                             font=("Segoe UI", 10))
    amount_entry.pack(pady=5)
    amount_entry.focus_set() # Set focus to entry field

    def apply_change():
        try:
            new_amount_str = amount_var.get().replace('€', '').strip()
            new_amount = float(new_amount_str)
            if new_amount < 0:
                 messagebox.showwarning("Invalid Input", "Bankroll cannot be negative.", parent=dialog)
                 return
                 
            global bankroll
            bankroll = new_amount
            save_bankroll(bankroll)
            br_label.config(text=f"Bankroll: €{bankroll:.2f}")
            feedback_label.config(text=f"Bankroll manually set to €{bankroll:.2f}")
            logging.info(f"Bankroll manually set to €{bankroll:.2f}")
            dialog.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number.", parent=dialog)

    # Buttons frame
    btn_frame = tk.Frame(dialog, bg="#313131")
    btn_frame.pack(pady=10)

    ttk.Button(btn_frame, text="Apply", command=apply_change, style="TButton").pack(side=tk.LEFT, padx=10)
    ttk.Button(btn_frame, text="Cancel", command=dialog.destroy, style="TButton").pack(side=tk.LEFT, padx=10)

    # Bind Enter key within the dialog
    dialog.bind('<Return>', lambda event: apply_change())

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

# <<< NEW HELPER FUNCTION for Original Format >>>
def _parse_format_original(lines):
    """Parses bet information assuming the original site format."""
    logging.info("Attempting parsing using Original Format rules...")
    bets = []
    i = 0
    num_lines = len(lines)
    
    # Patterns specific to the original format
    selected_bet_pattern = re.compile(r'^(.*?)\s+(\d+[,.]?\d*)\b.*?$', re.IGNORECASE)
    ignore_keywords = ["Stake", "To Return", "Cash Out", "Single", "Result", "Payout"]
    stake_label_pattern = re.compile(r'Stake', re.IGNORECASE)
    amount_value_pattern = re.compile(r'(\d+[.,]\d+)')
    date_pattern = re.compile(r'([A-Za-z]{3}\s+\d{1,2}\s+[A-Za-z]{3})', re.IGNORECASE)
    time_pattern = re.compile(r'(\d{2})[:.,]?(\d{2})$')
    # Add a pattern to find "Team A vs Team B" or "Team A - Team B"
    vs_pattern = re.compile(r'^\s*(.+?)\s+(?:vs[.]?|-)\s+(.+?)\s*$', re.IGNORECASE)

    # Parsing Loop (same as before)
    while i < num_lines:
        line = lines[i]
        logging.debug(f"[Original Format] Processing line {i}: '{line}'")
        processed_bet_on_this_line = False

        selected_match = selected_bet_pattern.match(line)
        if selected_match:
            potential_team_name = selected_match.group(1).strip()
            is_ignored = any(keyword.lower() in potential_team_name.lower() for keyword in ignore_keywords)
            
            if not is_ignored:
                team_name_from_bet_line = potential_team_name # Store the initially found name
                odds_str_raw = selected_match.group(2)
                try: 
                    # ... (Odds correction and validation logic remains the same) ...
                    odds_str = odds_str_raw.replace(',', '.')
                    odds = 0.0 
                    if '.' not in odds_str and odds_str.isdigit():
                        odds_int = int(odds_str)
                        if 101 <= odds_int <= 2000: 
                            odds = float(odds_int) / 100.0
                            logging.info(f"    [Original Format] Corrected integer odds: '{odds_str_raw}' -> {odds:.2f}")
                        else:
                            odds = float(odds_str) 
                    else:
                        odds = float(odds_str)
                    
                    MIN_ODDS = 1.01
                    MAX_ODDS = 20.0 
                    if not (MIN_ODDS <= odds <= MAX_ODDS):
                        logging.warning(f"    [Original Format] Odds {odds:.2f} ('{odds_str_raw}') invalid. Skipping.")
                        i += 1 # Move to next line if odds invalid
                        continue 
                    # --- End Odds Validation ---

                    # --- Attempt to find full matchup --- 
                    match_display_string = team_name_from_bet_line # Default to the single team name
                    # Look +/- 2 lines around the current bet line (i)
                    search_start = max(0, i - 2)
                    search_end = min(num_lines, i + 3)
                    for k in range(search_start, search_end):
                        vs_match = vs_pattern.match(lines[k])
                        if vs_match:
                            team_a = vs_match.group(1).strip()
                            team_b = vs_match.group(2).strip()
                            # Basic check: does one of the teams contain the betted team name?
                            # This helps avoid grabbing unrelated matchups nearby.
                            if team_name_from_bet_line.lower() in team_a.lower() or \
                               team_name_from_bet_line.lower() in team_b.lower():
                                match_display_string = f"{team_a} vs {team_b}"
                                logging.info(f"    [Original Format] Found full matchup '{match_display_string}' on line {k}.")
                                break # Found a likely match, stop searching
                    # --- End find full matchup ---
                    
                    # Use the potentially updated match_display_string
                    logging.info(f"[Original Format] Potential Bet: Match='{match_display_string}', Odds={odds:.2f}")
                    current_bet = {'team': match_display_string, 'odds': odds} # Store the full string
                    found_stake = False
                    found_date = False
                    found_time = False
                    last_scanned_line_idx = i
                    search_end_idx = min(i + 7, num_lines)
                    j = i + 1
                    while j < search_end_idx:
                        scan_line = lines[j]
                        last_scanned_line_idx = j
                        # Scan logic for stake, date, time
                        if not found_stake:
                            # This block needs to be indented
                            stake_match = stake_label_pattern.search(scan_line)
                            if stake_match:
                                # This block needs to be indented under `if stake_match:`
                                search_text_for_amount = scan_line[stake_match.end():].strip()
                                if not search_text_for_amount and (j + 1 < search_end_idx):
                                     search_text_for_amount = lines[j+1].strip()
                                     if amount_value_pattern.fullmatch(search_text_for_amount):
                                          last_scanned_line_idx = j + 1
                                     else:
                                          search_text_for_amount = ""
                                amount_match = amount_value_pattern.search(search_text_for_amount)
                                if amount_match:
                                    try:
                                        stake_str = amount_match.group(1).replace(',', '.')
                                        current_bet['stake'] = float(stake_str)
                                        found_stake = True
                                        logging.info(f"    [Original Format] Found Stake: {current_bet['stake']}")
                                    except ValueError: 
                                         logging.warning(f"    [Original Format] Failed parse stake: '{amount_match.group(1)}'")
                        # Indentation for date/time checks should align with the `if not found_stake:` check
                        if not found_date:
                            date_match = date_pattern.search(scan_line)
                            if date_match: current_bet['date'] = date_match.group(1); found_date = True; logging.info(f"    [Original Format] Found Date: {current_bet['date']}")
                        if not found_time:
                            time_match = time_pattern.search(scan_line)
                            if time_match: current_bet['time'] = f"{time_match.group(1)}:{time_match.group(2)}"; found_time = True; logging.info(f"    [Original Format] Found Time: {current_bet['time']}")
                        j += 1
                    
                    if found_stake:
                        current_bet['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Fallback
                        if found_date and found_time:
                            try:
                                formatted_date_str = f"{current_bet['date']} {datetime.now().year} {current_bet['time']}"
                                current_bet['datetime'] = datetime.strptime(formatted_date_str, '%a %d %b %Y %H:%M').strftime('%Y-%m-%d %H:%M:%S')
                            except Exception: pass
                        logging.info(f"[Original Format] Adding complete bet: {current_bet}")
                        bets.append(current_bet)
                        i = last_scanned_line_idx # Jump index
                        processed_bet_on_this_line = True
                    else:
                        logging.warning(f"[Original Format] Discarding bet (Stake missing): Match='{match_display_string}'")
                except ValueError: 
                    logging.warning(f"[Original Format] Could not parse odds '{odds_str_raw}'. Skipping.")
        
        if not processed_bet_on_this_line:
            i += 1
    
    logging.info(f"Original Format parsing finished. Found {len(bets)} bets.")
    return bets
# <<< END HELPER FUNCTION >>>

# <<< NEW HELPER FUNCTION for Friend Format >>>
def _parse_format_friend(lines):
    """Parses bet information assuming the friend's screenshot format."""
    logging.info("Attempting parsing using Friend Format rules...")
    bets = []
    num_lines = len(lines)
    i = 0

    # Regex Patterns for Friend Format
    # Selection line: Starts with ○, captures text before the last number (odds)
    # Allows for potential spaces around the circle symbol
    selection_pattern = re.compile(r'^\s*○\s*(.+?)\s+([\d.,]+)\s*$', re.IGNORECASE)
    # Stake line: Starts with Stake, captures the euro amount
    stake_pattern = re.compile(r'^\s*Stake\s+€?([\d.,]+)\s*$', re.IGNORECASE)

    while i < num_lines:
        line = lines[i]
        logging.debug(f"[Friend Format] Processing line {i}: '{line}'")
        processed_bet_on_this_line = False

        selection_match = selection_pattern.match(line)

        if selection_match:
            try:
                selection_text = selection_match.group(1).strip()
                odds_str_raw = selection_match.group(2)
                odds_str = odds_str_raw.replace(',', '.')
                odds = float(odds_str)
                
                # Validate Odds (adjust range if necessary)
                MIN_ODDS = 1.01
                MAX_ODDS = 50.0 # Increased max odds range slightly
                if not (MIN_ODDS <= odds <= MAX_ODDS):
                    logging.warning(f"    [Friend Format] Odds {odds:.2f} (from '{odds_str_raw}') invalid. Skipping partial bet.")
                    i += 1 
                    continue 

                logging.info(f"[Friend Format] Potential Bet: Selection='{selection_text}', Odds={odds:.2f}")
                current_bet = {'team': selection_text, 'odds': odds} # Use selection_text as 'team' for now
                found_stake = False
                last_scanned_line_idx = i
                
                # Look ahead a few lines for the Stake
                search_end_idx = min(i + 5, num_lines) # Look max 4 lines ahead
                j = i + 1
                while j < search_end_idx:
                    scan_line = lines[j]
                    last_scanned_line_idx = j # Keep track of how far we scanned
                    stake_match = stake_pattern.match(scan_line)
                    if stake_match:
                        try:
                            stake_str_raw = stake_match.group(1)
                            stake_str = stake_str_raw.replace(',', '.')
                            current_bet['stake'] = float(stake_str)
                            found_stake = True
                            logging.info(f"    [Friend Format] Found Stake: {current_bet['stake']:.2f}")
                            break # Stop searching once stake is found
                        except ValueError:
                            logging.warning(f"    [Friend Format] Failed to parse stake from '{stake_str_raw}' on line {j}")
                            # Continue searching in case of malformed number
                    j += 1

                if found_stake:
                    # Use current timestamp as fallback
                    current_bet['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    # Attempt to determine league (Can use fuzzy matching later if needed)
                    # For now, use the selection text or a default
                    current_bet['league'] = get_team_league(selection_text) # Try to guess league
                    if current_bet['league'] == "Other":
                         current_bet['league'] = "Unknown League" # Default if guess fails
                    logging.info(f"[Friend Format] Adding complete bet: {current_bet}")
                    bets.append(current_bet)
                    i = last_scanned_line_idx # Jump scan index past the stake line
                    processed_bet_on_this_line = True
                else:
                    logging.warning(f"[Friend Format] Discarding bet (Stake not found near selection): Selection='{selection_text}'")

            except ValueError:
                logging.warning(f"[Friend Format] Could not parse odds '{selection_match.group(2)}'. Skipping line.")
            except Exception as e:
                logging.error(f"[Friend Format] Unexpected error parsing bet starting line {i}: {e}", exc_info=True)

        if not processed_bet_on_this_line:
            i += 1 # Move to the next line if we didn't process a bet starting here
            
    logging.info(f"Friend Format parsing finished. Found {len(bets)} bets.")
    return bets
# <<< END HELPER FUNCTION >>>

# <<< NEW HELPER FUNCTION for New Site Format >>>
def _parse_format_new_site(lines):
    """Parses bet information assuming the new screenshot format provided."""
    logging.info("Attempting parsing using New Site Format rules...")
    bets = []
    num_lines = len(lines)
    i = 0

    # --- Regex Patterns for New Site ---
    # Update selection pattern to look for bullet • or other leading non-word char
    # Make the leading symbol optional/general to handle OCR variations
    selection_pattern = re.compile(r'^\s*\W?\s*(.+?)\s+([\d.,]+)\s*$', re.IGNORECASE) # Changed from ✔
    bet_type_pattern = re.compile(r'^(Double Chance|Match Result|Over/Under)', re.IGNORECASE) 
    stake_pattern = re.compile(r'^\s*Stake\s+€?([\d.,]+)\s*$', re.IGNORECASE)
    ignore_words_solo = ["return", "stake", "single", "double chance", "cagliari", "fiorentina", "genoa", "lazio", "to return", "cash out"]

    while i < num_lines:
        line = lines[i]
        logging.debug(f"[New Site Format] Processing line {i}: '{line}'")
        processed_bet_on_this_line = False

        selection_match = selection_pattern.match(line)

        if selection_match:
            try:
                selection_text = selection_match.group(1).strip()
                odds_str_raw = selection_match.group(2)
                odds_str = odds_str_raw.replace(',', '.')
                odds = float(odds_str)
                
                MIN_ODDS = 1.01
                MAX_ODDS = 50.0 
                if not (MIN_ODDS <= odds <= MAX_ODDS):
                    logging.warning(f"    [New Site] Odds {odds:.2f} invalid. Skipping.")
                    i += 1 
                    continue 

                logging.info(f"[New Site] Potential Bet: Selection='{selection_text}', Odds={odds:.2f}")
                current_bet = {'selection': selection_text, 'odds': odds}
                found_stake = False
                found_teams = False
                team_a = "Team A?"
                team_b = "Team B?"
                last_scanned_line_idx = i

                search_end_idx = min(i + 8, num_lines)
                # Remove dependency on finding bet type line first
                # found_bet_type_line_idx = -1 
                potential_teams = []
                
                # --- Find Stake and Potential Teams --- 
                # We need stake, and try to find 2 teams in the next few lines
                for j in range(i + 1, search_end_idx):
                    scan_line = lines[j].strip()
                    last_scanned_line_idx = max(last_scanned_line_idx, j)

                    # Check for Stake
                    if not found_stake:
                        stake_match = stake_pattern.match(scan_line)
                        if stake_match:
                            try:
                                stake_str = stake_match.group(1).replace(',', '.')
                                current_bet['stake'] = float(stake_str)
                                found_stake = True
                                logging.info(f"    [New Site] Found Stake: {current_bet['stake']:.2f} on line {j}")
                            except ValueError:
                                logging.warning(f"    [New Site] Failed parse stake: '{stake_match.group(1)}' on line {j}")
                                
                    # Look for Teams (if we haven't found 2 already)
                    if not found_teams:
                        # Use the same heuristic as before, but don't wait for bet_type line
                        if scan_line and re.search(r'[a-zA-Z]', scan_line) and \
                           scan_line.lower() not in ignore_words_solo and \
                           not stake_pattern.match(scan_line) and \
                           not selection_pattern.match(scan_line): 

                             cleaned_team_name = re.sub(r'(^\W+|\W+$)', '', scan_line).strip()
                             
                             if cleaned_team_name: 
                                  # Basic check: Avoid adding the bet selection text itself as a team name
                                  # Use SequenceMatcher for slightly fuzzy comparison
                                  selection_part = selection_text.split(' or ')[0].strip() # Get primary part of selection
                                  similarity = SequenceMatcher(None, cleaned_team_name.lower(), selection_part.lower()).ratio()
                                  
                                  if similarity < 0.8: # If it's not too similar to the selection
                                       potential_teams.append(cleaned_team_name)
                                       logging.debug(f"        [New Site] Potential team appended: '{cleaned_team_name}'")
                                       if len(potential_teams) == 2:
                                           team_a = potential_teams[0]
                                           team_b = potential_teams[1]
                                           found_teams = True
                                           logging.info(f"    [New Site] Found potential Teams: '{team_a}' and '{team_b}'")
                                  else:
                                       logging.debug(f"        [New Site] Skipping potential team '{cleaned_team_name}' - too similar to selection '{selection_part}'")

                    # If we found stake AND 2 teams, we can stop scanning early
                    if found_stake and found_teams:
                         break 

                # --- Finalize Bet --- (Logic remains the same)
                # ... existing code ...

                if found_stake:
                    if found_teams:
                         match_display_string = f"{team_a} vs {team_b}"
                    else:
                         match_display_string = re.sub(r'\s+or\s+Draw$', '', selection_text, flags=re.IGNORECASE).strip()
                         logging.warning(f"    [New Site] Teams not reliably found. Using '{match_display_string}' as Match name.")
                    
                    current_bet['team'] = match_display_string
                    current_bet['league'] = get_team_league(team_a if found_teams else match_display_string)
                    if current_bet['league'] == "Other": current_bet['league'] = "Unknown League"
                    current_bet['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] 
                    
                    logging.info(f"[New Site] Adding complete bet: {current_bet}")
                    bets.append(current_bet)
                    i = last_scanned_line_idx 
                    processed_bet_on_this_line = True
                else:
                    logging.warning(f"[New Site] Discarding bet (Stake missing): Selection='{selection_text}'")

            except ValueError:
                logging.warning(f"[New Site] Could not parse odds '{selection_match.group(2)}'. Skipping line.")
            except Exception as e:
                logging.error(f"[New Site] Unexpected error parsing bet line {i}: {e}", exc_info=True)

        if not processed_bet_on_this_line:
            i += 1
            
    logging.info(f"New Site Format parsing finished. Found {len(bets)} bets.")
    return bets
# <<< END HELPER FUNCTION >>>

def extract_bet_info(image_path):
    """Extracts bet info by detecting format and calling the appropriate parser."""
    if reader is None:
        logging.error("EasyOCR Reader not initialized.")
        feedback_label.config(text="OCR Engine Error (EasyOCR not initialized)")
        return None
        
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB': img = img.convert('RGB')
        img_np = np.array(img) 
        logging.debug(f"Processing image '{os.path.basename(image_path)}'")

        result = reader.readtext(img_np, detail=0, paragraph=True)
        text = "\n".join(result) 
        logging.info(f"--- EasyOCR Raw Output ---\n{text}\n--- End OCR Output ---")

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            logging.warning("OCR produced no text lines.")
            feedback_label.config(text="OCR found no text in the image.")
            return None

        # --- Format Detection Logic --- 
        detected_format = None
        bets = None # Initialize bets to None
        
        # --- Heuristics Check --- 
        text_lower = text.lower()
        lines_lower = [l.lower() for l in lines]
        
        # --- New Site Format Specifics (Checkmark or Bullet) ---
        new_site_selection_pattern = r'^\s*\W?\s*(.+?)\s+[\d.,]+\s*$' # Looser check for selection line
        has_new_site_selection = re.search(new_site_selection_pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None
        has_bet_type_line = any(re.match(r'^(double chance|match result|over/under)', l) for l in lines_lower)
        has_new_site_stake = re.search(r'^stake\s+€?[\d.,]+', text_lower, flags=re.MULTILINE) is not None
        new_site_format_likely = has_new_site_selection and has_bet_type_line and has_new_site_stake

        # --- Friend Format Specifics ---
        # Restore Friend Format Check
        friend_format_likely = False
        try:
            for i, line in enumerate(lines_lower):
                if line.strip().startswith('○'):
                    search_end = min(i + 5, len(lines_lower))
                    for j in range(i + 1, search_end):
                        if re.match(r'^\s*stake\s+€[\d.,]+', lines_lower[j]):
                            friend_format_likely = True
                            break
                    if friend_format_likely:
                        break
        except Exception as e:
             logging.warning(f"Error during Friend format heuristic check: {e}")
             friend_format_likely = False

        # --- Coolbet Format Specifics: ---
        # Restore Coolbet Format Check
        team_vs_team_pattern = r'^(.+?)\s+-\s+(.+?)$' 
        has_team_vs_team = re.search(team_vs_team_pattern, text, flags=re.MULTILINE) is not None 
        has_match_result_line = "match result (1x2)" in text_lower
        coolbet_format_likely = has_team_vs_team and has_match_result_line
        
        # --- Original Format Specifics (Less precise): ---
        has_original_stake_label = 'stake' in text_lower 

        logging.debug(f"Format detection heuristics: new_site_likely={new_site_format_likely}, friend_likely={friend_format_likely}, coolbet_likely={coolbet_format_likely}, has_original_stake={has_original_stake_label}")

        # --- Apply Detection Rules (Adjusted Order) ---
        # Prioritize the more specific new format
        if new_site_format_likely:
            detected_format = "New Site"
            bets = _parse_format_new_site(lines)
        elif friend_format_likely:
            detected_format = "Friend"
            bets = _parse_format_friend(lines)
        elif coolbet_format_likely:
            detected_format = "Coolbet"
            bets = _parse_format_coolbet(lines)
        elif has_original_stake_label:
            detected_format = "Original"
            bets = _parse_format_original(lines)
        else:
            # Fallback: Try all parsers if no specific format detected clearly
            logging.warning("Could not reliably detect format based on primary heuristics. Trying fallbacks...")
            # Try New Site first in fallback?
            detected_format = "New Site (Fallback)"
            bets = _parse_format_new_site(lines)
            if not bets:
                detected_format = "Friend (Fallback)"
                bets = _parse_format_friend(lines)
                if not bets:
                     detected_format = "Original (Fallback)"
                     bets = _parse_format_original(lines)
                     if not bets:
                          detected_format = "Coolbet (Fallback)"
                          bets = _parse_format_coolbet(lines)
        
        # --- Process results --- 
        if bets:
            logging.info(f"Successfully extracted {len(bets)} bets using {detected_format} parser.")
            # Update feedback based on number of bets found
            if len(bets) == 1:
                 feedback_text = f"Found 1 bet ({detected_format}):\n"
            else:
                 feedback_text = f"Found {len(bets)} bets ({detected_format}):\n"
            # Display details of each found bet
            for b in bets:
                 feedback_text += f"• {b.get('team','N/A')} ({b.get('odds',0):.2f}) | Stake: €{b.get('stake',0):.2f} | L: {b.get('league','?')} | T: {b.get('datetime', 'N/A')}\n" # Added League + Timestamp
            feedback_label.config(text=feedback_text.strip()) # Remove trailing newline
            return bets
        else:
            # Log failure and show preview
            logging.warning(f"No valid bets found using {detected_format} parser(s).")
            preview = "\n".join(lines[:15]) # Show first 15 lines
            feedback_label.config(text=f"No valid bets found. Parser: {detected_format}. Preview:\n{preview}")
            return None
            
    # ... (rest of the except blocks remain the same) ...
    except AttributeError as ae:
         # ...
         # Catch the specific AttributeError we saw
         logging.error(f"AttributeError during EasyOCR processing (possibly image loading issue): {ae}")
         import traceback
         logging.error(traceback.format_exc())
         feedback_label.config(text=f"Error processing image format/data. Check logs.")
         return None
    except Exception as e:
        # Add pass to fix indentation error
        pass 
        # ... (Existing exception handling code) ...
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"Error in extract_bet_info (EasyOCR): {e}\n{error_details}")
        # Check if it's an EasyOCR model download issue
        if "download attempt failed" in str(e).lower():
             messagebox.showerror("EasyOCR Error", "Failed to download EasyOCR models. Check internet connection and permissions.")
             feedback_label.config(text="EasyOCR Model Download Failed.")
        else:
            feedback_label.config(text=f"Error processing image with EasyOCR: {str(e)}. Check logs.")
        return None

def process_file(file_path):
    if not file_path:
        return
    extracted_bets = extract_bet_info(file_path) # Renamed variable
    if extracted_bets:
        # Now handle the list of extracted bets
        num_added = 0
        num_updated = 0
        base_timestamp = datetime.now() # Get a base time for this batch
        
        for i, bet in enumerate(extracted_bets):
            # Prepare data for add_bet or update_result
            match_name_raw = bet.get('team', 'Unknown Match') 
            odds = bet.get('odds', 0.0)
            stake = bet.get('stake', 0.0)
            # Use extracted datetime OR increment base timestamp for ordering
            bet_datetime_str = bet.get('datetime', 
                                         (base_timestamp + timedelta(milliseconds=i*10)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]) # Add ms

            # --- Clean the extracted match name --- 
            match_name_cleaned = match_name_raw
            if isinstance(match_name_raw, str):
                # 1. Remove common prefixes (case-insensitive)
                prefixes_to_remove = [r'^o\s+', r'^\.\s+', r'^Draw\s+or\s+'] # Add more prefixes if needed
                for prefix_pattern in prefixes_to_remove:
                     match_name_cleaned = re.sub(prefix_pattern, '', match_name_cleaned, flags=re.IGNORECASE).strip()
                     if match_name_cleaned != match_name_raw:
                          logging.debug(f"Removed prefix '{prefix_pattern}' from '{match_name_raw}' -> '{match_name_cleaned}'")
                          match_name_raw = match_name_cleaned # Update for next potential prefix removal
                
                # 2. Remove " or Draw" suffix (case-insensitive)
                suffix_pattern = r'\s+or\s+Draw$'
                original_name = match_name_cleaned # Store before suffix removal
                match_name_cleaned = re.sub(suffix_pattern, '', match_name_cleaned, flags=re.IGNORECASE).strip()
                if match_name_cleaned != original_name:
                    logging.debug(f"Removed ' or Draw' suffix from '{original_name}' -> '{match_name_cleaned}'")
            # --- End Cleaning ---

            # Decision logic (Add vs Update) - Use the cleaned name and new timestamp
            add_bet(match_name_cleaned, odds, stake, bet_datetime_str) 
            num_added += 1
            
            # ... rest of loop ...

        feedback_label.config(text=f"Processed screenshot: Added {num_added} bet(s).") # Update feedback
        logging.info(f"Processed screenshot {os.path.basename(file_path)}: Added {num_added} bets.")

    else:
        # Keep existing feedback for extraction failure
        feedback_label.config(text="Failed to extract valid bet info. Check screenshot/log.")
        logging.warning(f"Failed to extract bet info from {os.path.basename(file_path)}.")


# Modify add_bet to accept the extracted datetime
def add_bet(match, odds, stake, bet_datetime_str):
    global history
    league = get_team_league(match) 
    
    # Ensure stake and odds are floats
    try:
        stake_float = float(stake)
        odds_float = float(odds)
    except (ValueError, TypeError):
        logging.error(f"Invalid stake ({stake}) or odds ({odds}) for match {match}. Skipping bet.")
        feedback_label.config(text=f"Error adding {match}: Invalid stake or odds.")
        return # Don't add the bet if data is invalid

    # Create DataFrame for the new bet
    new_bet = pd.DataFrame({
        "Match": [match],
        "League": [league],
        "Odds": [odds_float],
        "Stake": [stake_float],
        "Result": ["Pending"],
        "Payout": [0.0], 
        # Use the provided datetime string
        "Date": [bet_datetime_str] 
    })
    
    # Check if an identical pending bet already exists
    is_duplicate = False
    if not history.empty:
       pending_bets = history[history["Result"] == "Pending"]
       duplicates = pending_bets[
            (pending_bets["Match"] == match) &
            (pending_bets["Odds"].round(2) == round(odds_float, 2)) &
            (pending_bets["Stake"].round(2) == round(stake_float, 2)) 
            # Optionally check date/time proximity if available and reliable
       ]
       if not duplicates.empty:
           is_duplicate = True
           logging.warning(f"Duplicate pending bet detected for {match}. Skipping.")
           feedback_label.config(text=f"Skipped duplicate: {match}")

    if not is_duplicate:
        # Concatenate using pd.concat
        history = pd.concat([history, new_bet], ignore_index=True)
        save_history(history)
        update_history_table() 
        update_league_filter() # Update league list in case of new league
        feedback_label.config(text=f"Added: {match} ({league})\nOdds: {odds:.2f}, Stake: €{stake:.2f}")
        logging.info(f"Added bet: {match} ({league}), Odds: {odds:.2f}, Stake: €{stake:.2f}, Date: {bet_datetime_str}")
    
# Modify update_history_table to handle the potentially more precise datetime
def update_history_table(filtered_df=None, show_only_today: bool = False):
    """Updates the Treeview with bet history, applying styles and formatting.

    Args:
        filtered_df (pd.DataFrame, optional): A pre-filtered DataFrame (e.g., by league).
                                             If provided, show_only_today is ignored. Defaults to None.
        show_only_today (bool, optional): If True and filtered_df is None, only show bets from today.
                                          Defaults to False.
    """
    for item in tree.get_children():
        tree.delete(item)

    # Determine the base DataFrame to display
    base_df = history.copy() # Start with the full history

    # Apply filtering: League filter takes precedence
    display_df = None
    if filtered_df is not None:
        display_df = filtered_df # Use the league-filtered data
        logging.info("Displaying league-filtered data in Treeview.")
    elif show_only_today: # Apply daily filter only if no league filter and flag is True
        logging.info("Attempting to filter Treeview for today's bets...")
        try:
            # Ensure Date_dt column exists or create it for filtering
            if 'Date_dt' not in base_df.columns:
                if 'Date' in base_df.columns:
                    # Use the correct format directly
                    base_df['Date_dt'] = pd.to_datetime(base_df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                    # Remove the fallback to mixed format as we know the expected one
                    # if base_df['Date_dt'].isnull().any():
                    #     logging.warning("Failed to parse some dates with standard format, trying mixed.")
                    #     base_df['Date_dt'] = pd.to_datetime(base_df['Date'], errors='coerce', format='mixed')

                    if base_df['Date_dt'].isnull().all():
                        logging.error("Created 'Date_dt' column, but all values are NaT after parsing. Cannot filter by day.")
                    else:
                         logging.debug("Created 'Date_dt' column for daily filtering.")
                else:
                    logging.error("Cannot filter by day: 'Date'/'Date_dt' columns missing.")
                    # Proceed without filtering if no date column exists

            # Apply the filter if Date_dt is usable
            if 'Date_dt' in base_df.columns and not base_df['Date_dt'].isnull().all():
                today_date = pd.Timestamp.now().normalize() # Get today's date (midnight)
                logging.debug(f"Filtering for date: {today_date}")
                # Keep rows where the date part of Date_dt matches today
                mask = base_df['Date_dt'].dt.normalize() == today_date
                display_df = base_df[mask].copy() # Use .copy() to avoid SettingWithCopyWarning
                logging.info(f"Filtered Treeview to show {len(display_df)} bets from {today_date.strftime('%Y-%m-%d')}.")
                if display_df.empty:
                    logging.info("No bets found for today.")
            else:
                # Logged errors above if Date_dt couldn't be created or used
                logging.warning("Proceeding without daily filter due to missing or unparseable date data.")
                display_df = base_df # Show all if filtering failed

        except Exception as e:
            logging.error(f"Error during daily filtering in update_history_table: {e}. Displaying unfiltered data for safety.")
            display_df = base_df # Show all as a fallback
    else:
        # No league filter and show_only_today is False
        display_df = base_df # Show all history
        logging.info("Displaying all bets (no league filter, show_only_today=False).")

    # If after all filtering, display_df is still None (shouldn't happen ideally), default to base_df
    if display_df is None:
        logging.warning("display_df was None unexpectedly, defaulting to full history.")
        display_df = base_df

    # --- Safely create Date_dt for sorting ---
    if 'Date' in display_df.columns:
        try:
            # Use the correct format directly for sorting as well
            date_dt_col = pd.to_datetime(display_df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            # Remove the try/except for millisecond format
            # try:
            #     date_dt_col = pd.to_datetime(display_df['Date'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
            #     logging.debug("Parsed Date column with milliseconds format.")
            # except (ValueError, TypeError):
            #      logging.debug("Millisecond format failed, trying mixed format for Date column.")
            #      date_dt_col = pd.to_datetime(display_df['Date'], errors='coerce', format='mixed')
                 
            if not date_dt_col.isnull().all():
                display_df['Date_dt'] = date_dt_col
                sort_columns = ['Date_dt'] 
                sort_ascending = [False] 
                logging.debug("Sorting history table primarily by Date_dt (desc).") # Simplified log
            else:
                 logging.warning("Could not parse any dates in 'Date' column for sorting. Order might be unpredictable.")
        except Exception as e:
            logging.warning(f"Error processing 'Date' column for sorting: {e}. Order might be unpredictable.")
    else:
        logging.warning("No 'Date' column found for sorting. Order might be unpredictable.")
        
    # Sort the DataFrame based ONLY on Date_dt if available
    try:
        if sort_columns: # Only sort if we have a valid date column
            display_df = display_df.sort_values(by=sort_columns, ascending=sort_ascending)
        else:
            # If no date column, maybe sort by index to retain add order? Risky.
            # For now, leave unsorted if no date.
            logging.warning("Displaying history potentially unsorted due to missing/unparsable Date column.")
    except Exception as e:
        logging.error(f"Unexpected error during sorting: {e}")

    # Add rows to Treeview
    for index, row in display_df.iterrows():
        # --- Get data from the DataFrame row FIRST --- 
        result = row.get("Result", "Pending")
        stake = row.get("Stake", 0.0)
        payout = row.get("Payout", 0.0)
        odds = row.get("Odds", 0.0)
        match_name = row.get("Match", "N/A")
        league = row.get("League", "N/A")
        date_str = row.get("Date", "N/A")
        df_index = index # Get the DataFrame index for this row
        logging.debug(f"Processing row for Treeview, DataFrame index: {df_index}, Match: {match_name}")
        # --- END Get data ---

        display_date = date_str # Default display
        try: # Format date for display if possible
             dt_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
             display_date = dt_obj.strftime('%d %b %Y %H:%M')
        except (ValueError, TypeError): 
             try: 
                 dt_obj = datetime.strptime(date_str, '%d.%m.%Y')
                 display_date = dt_obj.strftime('%d %b %Y')
             except (ValueError, TypeError):
                 pass 
        
        # Determine tags based on result and prepare action symbols
        tags = []
        win_symbol = "" # Blank by default
        loss_symbol = "" # Blank by default
        delete_symbol = "🗑️" # Always show delete symbol
        
        if result == "Pending":
            tags.append('pending')
            payout_str = "€0.00"
            win_symbol = "✅" # Show Win symbol for pending
            loss_symbol = "❌" # Show Loss symbol for pending
        elif result == "Win":
            tags.append('win')
            payout_str = f"+€{payout:.2f}" if payout > 0 else f"+€{stake * odds:.2f}"
        elif result == "Loss":
            tags.append('loss')
            payout_str = f"-€{stake:.2f}"
        else: 
             tags.append('pending') # Default tag
             payout_str = f"€{payout:.2f}"

        # Format values for display
        odds_str = f"{odds:.2f}"
        stake_str = f"€{stake:.2f}"
        
        # Insert item with NEW columns
        item_id = tree.insert("", "end", values=(
            match_name, league, odds_str, stake_str, result,
            payout_str, display_date, 
            win_symbol,  # New Win column value
            loss_symbol, # New Loss column value
            delete_symbol, # Delete column value
            df_index # Hidden DataFrame index
        ), tags=tuple(tags)) # Apply initial tags

        # Apply striping 
        status_tag = tags[0] if tags else 'pending' 
        final_tags = list(tags)
        if tree.index(item_id) % 2 == 1:
             final_tags.append('striped')
        # Apply the combined tags     
        tree.item(item_id, tags=tuple(final_tags)) 

# --- Treeview Action Functions ---
# Define these before they are bound to the treeview

def update_bet_result(selected_item_id, result):
    """Marks the selected bet as Win/Loss and updates bankroll/history."""
    logging.debug(f"update_bet_result called for item: {selected_item_id}, result: {result}") # Log entry
    global history, bankroll
    try:
        item_values = tree.item(selected_item_id)['values']
        logging.debug(f"  Item values retrieved: {item_values}") # Log values
        # Get data based on current column order: Match, League, Odds, Stake, Result, Payout, Date
        match_name = item_values[0]
        odds = float(item_values[2])
        stake_str = item_values[3].replace('€', '')
        stake = float(stake_str)
        current_result = item_values[4]
        
        if current_result != "Pending":
            logging.warning("Attempted to update non-pending bet.")
            messagebox.showwarning("Update Error", "Can only update result for Pending bets.")
            return

        # Find the bet in history
        matching_bets = history[
            (history["Match"] == match_name) &
            (history["Odds"].round(2) == round(odds, 2)) &
            (history["Stake"].round(2) == round(stake, 2)) &
            (history["Result"] == "Pending")
        ]
        logging.debug(f"  DataFrame search for pending bet returned empty: {matching_bets.empty}") # Log search result
        
        if matching_bets.empty:
             messagebox.showerror("Update Error", f"Could not find unique pending bet for {match_name}. Update manually if needed.")
             logging.error(f"Could not find unique pending bet in history for tree item: {item_values}")
             return
             
        idx = matching_bets.index[0] # Get index of the first match
        
        # Update result and bankroll
        payout = 0.0
        if result == "Win":
            payout = stake * odds
            bankroll += payout # Add full payout (stake was already accounted for implicitly)
            history.loc[idx, "Payout"] = payout
            logging.info(f"Marking Won: +€{payout:.2f} added to bankroll.")
        else: # Loss
            bankroll -= stake # Deduct stake
            history.loc[idx, "Payout"] = 0.0
            logging.info(f"Marking Lost: -€{stake:.2f} deducted from bankroll.")
        
        history.loc[idx, "Result"] = result
        save_bankroll(bankroll)
        save_history(history)
        br_label.config(text=f"Bankroll: €{bankroll:.2f}")
        
        update_history_table(show_only_today=True) # <<< Refresh main table showing today's bets
        feedback_label.config(text=f"Updated: {match_name} - {result}. Bankroll: €{bankroll:.2f}")
        logging.info(f"Updated bet result: {match_name} - {result}. New Bankroll: €{bankroll:.2f}")
        update_stats_display() # Update stats page as well
    except ValueError as e:
        messagebox.showerror("Update Error", f"Error processing values: {e}\nPlease check bet details.")
        logging.error(f"ValueError updating bet result: {e}")
    except IndexError:
        messagebox.showerror("Update Error", "Could not find the bet in the history data. It might have been deleted or modified unexpectedly.")
        logging.error(f"IndexError: Could not find the bet corresponding to tree item {selected_item_id} in the history DataFrame.")
    except Exception as e:
        messagebox.showerror("Update Error", f"An unexpected error occurred: {e}")
        logging.exception(f"Unexpected error updating bet result for item {selected_item_id}: {e}")

def delete_selected_bet(selected_item_id):
    """Deletes the selected bet from the Treeview and history."""
    logging.debug(f"delete_selected_bet called for item: {selected_item_id}")
    # --- Defensive global access for UI constants (in case of weird scope issue in exception) ---
    global CARD_BG_COLOR, BG_COLOR, TEXT_COLOR, ACCENT_COLOR, WIN_COLOR, LOSS_COLOR, TEXT_MUTED_COLOR # Added more potential constants
    # ---
    global history
    try:
        item_values = tree.item(selected_item_id)['values']
        # Make sure we get the DataFrame index stored as the *last* value
        # Check if length is less than 11 (since index is the 11th item, index 10)
        if len(item_values) < 11: 
            logging.error(f"Cannot delete: DataFrame index missing or item_values too short: {item_values}")
            messagebox.showerror("Delete Error", "Internal error: Cannot identify row to delete (invalid item values).")
            return
        
        # Access the 11th element (index 10) for the DataFrame index
        df_index_to_delete = item_values[10] 
        logging.debug(f"Attempting to delete row with DataFrame index: {df_index_to_delete}")

        # --- Convert extracted index to integer for comparison with DataFrame index ---
        # The DataFrame index is usually integer-based.
        try:
            df_index_to_delete = int(df_index_to_delete)
        except (ValueError, TypeError) as e:
            logging.error(f"Delete Error: Could not convert extracted index '{df_index_to_delete}' to an integer: {e}")
            messagebox.showerror("Delete Error", f"Internal error: Invalid index type '{type(df_index_to_delete).__name__}' found.")
            return
        # --- End Conversion ---

        if df_index_to_delete not in history.index:
             logging.error(f"Delete Error: Index {df_index_to_delete} not found in history DataFrame indices: {history.index.tolist()}")
             # Maybe refresh and see if it disappears?
             update_history_table(show_only_today=True)
             messagebox.showerror("Delete Error", "Could not find the selected bet in the history data. It might have already been deleted.")
             return

        # Get bet details *before* deleting for feedback
        deleted_row = history.loc[df_index_to_delete]
        match_name = deleted_row.get('Match', 'N/A')
        stake = deleted_row.get('Stake', 0.0)
        result = deleted_row.get('Result', 'Pending')

        # --- Bankroll Adjustment Logic --- 
        # ... (Bankroll adjustment logic remains the same) ...
        if result == "Win":
            payout = deleted_row.get("Payout", 0.0)
            try:
                 payout_num = float(payout)
                 adjust_bankroll(-payout_num) # Subtract the payout from bankroll
                 logging.info(f"Deleting WON bet. Bankroll adjusted by -€{payout_num:.2f}")
            except ValueError:
                 logging.warning(f"Could not parse payout '{payout}' for bankroll adjustment on deleting won bet.")
        elif result == "Loss":
             logging.info("Deleting LOST bet. No bankroll adjustment needed.")
        elif result == "Pending":
            try:
                 stake_num = float(stake)
                 adjust_bankroll(stake_num)
                 logging.info(f"Deleting PENDING bet. Stake €{stake_num:.2f} added back to bankroll.")
            except ValueError:
                 logging.warning(f"Could not parse stake '{stake}' for bankroll adjustment on deleting pending bet.")
        else:
             logging.warning(f"Deleting bet with unknown result '{result}'. Bankroll not adjusted.")

        # Drop the row from the DataFrame using the direct index
        history = history.drop(df_index_to_delete)
        logging.info(f"Successfully dropped row with index {df_index_to_delete} from history DataFrame.")
        
        # Save the updated history
        save_history(history)
        
        # Refresh the Treeview (showing today's bets on the main page)
        update_history_table(show_only_today=True) # <<< Refresh main table showing today's bets
        update_league_filter() # Update league list in case the deleted bet was the last of its league
        
        feedback_label.config(text=f"Deleted bet: {match_name}")
        logging.info(f"Deleted bet: {match_name} (Index: {df_index_to_delete})")
        
        # Update stats display as deletion affects stats
        update_stats_display()

    except IndexError:
        # This might happen if the item was already removed from the tree somehow
        logging.warning(f"IndexError retrieving item values for {selected_item_id}. Item might no longer exist.")
        # Attempt a refresh just in case
        update_history_table(show_only_today=True)
    except Exception as e:
        # This is the block where the error likely originates if messagebox is involved
        logging.exception(f"Unexpected error deleting bet for item {selected_item_id}: {e}")
        try:
            # Explicitly try showing the messagebox after potential error
            messagebox.showerror("Delete Error", f"An unexpected error occurred: {e}")
        except Exception as me:
             # Log if even the messagebox fails
             logging.error(f"CRITICAL: Failed to show error messagebox during delete: {me}")

def on_tree_click(event):
    """Handles left-clicks on the Treeview based on the clicked column."""
    logging.info(f"*** on_tree_click function CALLED at ({event.x}, {event.y}) ***")
    
    region = tree.identify("region", event.x, event.y)
    column_id = tree.identify_column(event.x)
    selected_item_id = tree.identify_row(event.y)
    logging.debug(f"  Identified region: {region}, item: {selected_item_id}, column_id: {column_id}")
    
    if region == "cell" and selected_item_id and column_id:
        try:
             # Get the 0-based index from the column ID (e.g., #8 -> 7)
             column_index = int(column_id.replace('#', '')) - 1
             logging.debug(f"  Calculated column index: {column_index}")
        except ValueError:
             logging.warning("  Could not determine column index from column_id.")
             return
             
        # --- Get item values --- 
        item_values = None
        try:
            item_values = tree.item(selected_item_id)['values']
            logging.debug(f"  Retrieved item values for {selected_item_id}: {item_values}")
        except Exception as e:
             logging.error(f"  Failed to get item values for {selected_item_id}: {e}")
             return # Cannot proceed without item values
        
        if not item_values or len(item_values) < 11: # Check length for new column count
            logging.warning(f"  Item values list invalid (None, empty, or too short) for {selected_item_id}. Aborting click action.")
            return

        # --- Determine Action Based on Clicked Column Index --- 
        # IMPORTANT: Adjust these indices if you reorder columns in the setup
        win_column_index = 7  # Index of "Win ✅"
        loss_column_index = 8 # Index of "Loss ❌"
        delete_column_index = 9 # Index of "Delete 🗑️"

        current_result = item_values[4] # Result is at index 4

        if column_index == delete_column_index:
            logging.info(f"--> Delete column ({column_index}) clicked for item: {selected_item_id}")
            delete_selected_bet(selected_item_id)
            return

        elif column_index == win_column_index:
            logging.debug(f"  Win column ({column_index}) clicked for item {selected_item_id}.")
            if current_result == "Pending":
                logging.info(f"    Marking bet as WIN. Calling update_bet_result(Win).")
                update_bet_result(selected_item_id, "Win")
            else:
                logging.debug(f"    Ignoring click on Win column because result is '{current_result}'.")
            return
            
        elif column_index == loss_column_index:
            logging.debug(f"  Loss column ({column_index}) clicked for item {selected_item_id}.")
            if current_result == "Pending":
                logging.info(f"    Marking bet as LOSS. Calling update_bet_result(Loss).")
                update_bet_result(selected_item_id, "Loss")
            else:
                logging.debug(f"    Ignoring click on Loss column because result is '{current_result}'.")
            return

        else:
            # Log clicks in other columns (Match, League, etc.)
            logging.debug(f"  Click was in non-actionable column (Index: {column_index}). No action taken.")
            # You could add selection logic here if desired

def show_context_menu(event):
    """Displays a right-click context menu for the selected Treeview item."""
    logging.debug(f"show_context_menu triggered at ({event.x_root}, {event.y_root})") 
    selected_item_id = tree.identify_row(event.y)
    logging.debug(f"  Identified item for context menu: {selected_item_id}")
    if selected_item_id:
        tree.selection_set(selected_item_id) # Select the row under the cursor
        menu = tk.Menu(root, tearoff=0, bg="#424242", fg="#E0E0E0", 
                       activebackground="#616161", activeforeground="#FFFFFF")
        
        # Add Edit Option
        menu.add_command(label="Edit Bet", 
                         command=lambda id=selected_item_id: (logging.debug(f"Context menu: Calling open_edit_bet_dialog({id})"), open_edit_bet_dialog(id)))
        menu.add_separator()
        # Existing Delete Option
        menu.add_command(label="Delete Bet", 
                        command=lambda id=selected_item_id: (logging.debug(f"Context menu: Calling delete_selected_bet({id})"), delete_selected_bet(id)))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

# --- Function to open Edit Bet Dialog --- 
def open_edit_bet_dialog(selected_item_id):
    """Opens a dialog window to edit the selected bet."""
    logging.debug(f"Opening edit dialog for item: {selected_item_id}")
    global history
    try:
        item_values = tree.item(selected_item_id)['values']
        # Ensure DataFrame index is present and valid
        if len(item_values) < 11: 
             logging.error(f"Cannot edit: DataFrame index missing or item_values too short: {item_values}")
             messagebox.showerror("Edit Error", "Internal error: Cannot identify row to edit.")
             return
             
        df_index = item_values[10] # Get the DataFrame index
        df_index = int(df_index) # Convert to integer

        if df_index not in history.index:
             logging.error(f"Edit Error: Index {df_index} not found in history DataFrame for item {selected_item_id}.")
             messagebox.showerror("Edit Error", "Could not find the selected bet in the data. It might have been deleted.")
             update_history_table(show_only_today=True) # Refresh table
             return

        # Get current data directly from the DataFrame for accuracy
        current_data = history.loc[df_index].to_dict()
        logging.debug(f"  Current data for index {df_index}: {current_data}")

    except (ValueError, TypeError, KeyError, IndexError) as e:
        logging.error(f"Error retrieving data for edit dialog (item: {selected_item_id}): {e}")
        messagebox.showerror("Edit Error", "Could not retrieve bet data. Please try again.")
        return

    # --- Create Dialog Window --- 
    dialog = tk.Toplevel(root)
    dialog.title("Edit Bet")
    dialog.configure(bg="#313131") 
    dialog.resizable(False, False)
    dialog.transient(root) 
    dialog.grab_set()

    # Center dialog (Helper function or inline code needed)
    # Simplified centering for now:
    root.update_idletasks()
    dialog.update_idletasks()
    # Define approximate size BEFORE calculating position
    dialog_width_approx = 450 
    dialog_height_approx = 350 
    x = root.winfo_x() + (root.winfo_width() // 2) - (dialog_width_approx // 2)
    y = root.winfo_y() + (root.winfo_height() // 2) - (dialog_height_approx // 2)
    dialog.geometry(f'{dialog_width_approx}x{dialog_height_approx}+{x}+{y}') 

    # --- Dialog Content --- 
    content_frame = tk.Frame(dialog, bg="#313131", padx=15, pady=15)
    content_frame.pack(expand=True, fill="both")
    content_frame.grid_columnconfigure(1, weight=1) # Allow entry column to expand

    # Variables to hold entry data
    entry_vars = {
        'match': tk.StringVar(value=current_data.get('Match', '')),
        'league': tk.StringVar(value=current_data.get('League', 'Other')),
        'odds': tk.StringVar(value=str(current_data.get('Odds', ''))),
        'stake': tk.StringVar(value=f"€{current_data.get('Stake', 0.0):.2f}"),
        'result': tk.StringVar(value=current_data.get('Result', 'Pending')),
        'date': tk.StringVar(value=current_data.get('Date', '')) # Store raw date string
    }

    row_idx = 0
    # Match
    ttk.Label(content_frame, text="Match:", background="#313131", foreground="#E0E0E0").grid(row=row_idx, column=0, sticky="e", pady=5, padx=5)
    match_entry = ttk.Entry(content_frame, textvariable=entry_vars['match'], width=40)
    match_entry.grid(row=row_idx, column=1, sticky="ew", pady=5, padx=5)
    row_idx += 1
    
    # League
    ttk.Label(content_frame, text="League:", background="#313131", foreground="#E0E0E0").grid(row=row_idx, column=0, sticky="e", pady=5, padx=5)
    # Populate league options dynamically
    known_leagues = sorted(list(set(l for l in history['League'].unique() if pd.notna(l) and l and l != 'Other'))) # Get unique known leagues from history
    league_options = ["Other"] + known_leagues 
    current_league = current_data.get('League', 'Other')
    if current_league not in league_options:
        league_options.insert(1, current_league) # Add current if unknown
    league_combo = ttk.Combobox(content_frame, textvariable=entry_vars['league'], values=league_options, state="normal", width=38) # Allow typing
    league_combo.grid(row=row_idx, column=1, sticky="ew", pady=5, padx=5)
    row_idx += 1
    
    # Odds
    ttk.Label(content_frame, text="Odds:", background="#313131", foreground="#E0E0E0").grid(row=row_idx, column=0, sticky="e", pady=5, padx=5)
    odds_entry = ttk.Entry(content_frame, textvariable=entry_vars['odds'], width=10)
    odds_entry.grid(row=row_idx, column=1, sticky="w", pady=5, padx=5)
    row_idx += 1
    
    # Stake
    ttk.Label(content_frame, text="Stake:", background="#313131", foreground="#E0E0E0").grid(row=row_idx, column=0, sticky="e", pady=5, padx=5)
    stake_entry = ttk.Entry(content_frame, textvariable=entry_vars['stake'], width=10)
    stake_entry.grid(row=row_idx, column=1, sticky="w", pady=5, padx=5)
    row_idx += 1
    
    # Date (Simple Entry for now)
    ttk.Label(content_frame, text="Date:", background="#313131", foreground="#E0E0E0").grid(row=row_idx, column=0, sticky="e", pady=5, padx=5)
    date_entry = ttk.Entry(content_frame, textvariable=entry_vars['date'], width=25)
    date_entry.grid(row=row_idx, column=1, sticky="w", pady=5, padx=5)
    row_idx += 1
    
    # Result
    ttk.Label(content_frame, text="Result:", background="#313131", foreground="#E0E0E0").grid(row=row_idx, column=0, sticky="e", pady=5, padx=5)
    result_options = ["Pending", "Win", "Loss"]
    result_combo = ttk.Combobox(content_frame, textvariable=entry_vars['result'], values=result_options, state="readonly", width=10)
    result_combo.grid(row=row_idx, column=1, sticky="w", pady=5, padx=5)
    row_idx += 1

    # --- Buttons --- 
    button_frame = tk.Frame(dialog, bg="#313131")
    button_frame.pack(pady=10)
    
    # Pass necessary info to save function (Placeholder for now)
    save_button = ttk.Button(button_frame, text="Save Changes", style="TButton", 
                             command=lambda: messagebox.showinfo("Not Implemented", "Save functionality not yet added.", parent=dialog)) # Placeholder command
                             # command=lambda d=dialog, idx=df_index, ev=entry_vars: save_edited_bet(d, idx, ev))
    save_button.pack(side=tk.LEFT, padx=10)
    
    cancel_button = ttk.Button(button_frame, text="Cancel", style="TButton", command=dialog.destroy)
    cancel_button.pack(side=tk.LEFT, padx=10)

    dialog.wait_window() # Wait for the dialog to be closed

# --- Functionality for Stats Page ---
# Define these functions *before* they are used in button commands

def show_main_page():
    """Hides the stats frame and shows the main frames."""
    logging.debug("Switching to Main Page")
    # Hide stats frame
    stats_frame.grid_remove()
    # Show main frames (using their original grid configuration)
    br_frame.grid(row=0, column=1, sticky="ew", padx=20, pady=(10, 5))
    feedback_frame.grid(row=1, column=1, sticky="ew", padx=20, pady=5)
    league_filter_frame.grid(row=2, column=1, sticky="ew", padx=20, pady=5)
    history_frame.grid(row=3, column=1, sticky="nsew", padx=20, pady=(5, 10))
    # Ensure history is updated (it might be filtered differently on stats page)
    update_history_table(show_only_today=True)

def update_stats_display(period='all'):
    """Calculates and updates the statistics labels on the stats page based on the selected period."""
    logging.info(f"--- update_stats_display entered (period: {period}) ---")
    global history, stats_labels, stats_period_title, TEXT_COLOR, WIN_COLOR, LOSS_COLOR # Ensure colors are accessible

    # --- Date Parsing and Filtering Logic (Keep as is from previous step) ---
    if 'Date' not in history.columns:
        logging.error("History DataFrame is missing 'Date' column. Cannot filter by period or calculate stats.")
        for key, label in stats_labels.items():
            label.config(text=f"{label.cget('text').split(':')[0]}: Error - No Date", foreground=LOSS_COLOR)
        stats_period_title.config(text="Statistics Error: Missing Date Column")
        return
    try:
        # Use the correct format directly
        history_dt = pd.to_datetime(history['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        # Remove the millisecond check and fallback
        # if history_dt.isnull().all(): 
        #     logging.warning("Could not parse any dates with milliseconds format, trying mixed format.")
        #     history_dt = pd.to_datetime(history['Date'], errors='coerce', format='mixed')
            
        if history_dt.isnull().any():
            logging.warning(f"Could not parse {history_dt.isnull().sum()} date(s) in history. These rows will be excluded from time-based stats.")
            
        df_analyzable = history.copy()
        df_analyzable['Date_dt'] = history_dt
        df_analyzable.dropna(subset=['Date_dt'], inplace=True)
    except Exception as e:
        logging.error(f"Error converting 'Date' column to datetime: {e}", exc_info=True)
        for key, label in stats_labels.items():
            label.config(text=f"{label.cget('text').split(':')[0]}: Error - Date Conv.", foreground=LOSS_COLOR)
        stats_period_title.config(text="Statistics Error: Date Conversion Failed")
        return

    now = pd.Timestamp.now()
    df_filtered = pd.DataFrame()
    period_text = "All Time"
    try:
        if period == 'today':
            start_date = now.normalize()
            df_filtered = df_analyzable[df_analyzable['Date_dt'] >= start_date]
            period_text = "Today"
        elif period == '7days':
            start_date = (now - timedelta(days=7)).normalize()
            df_filtered = df_analyzable[df_analyzable['Date_dt'] >= start_date]
            period_text = "Last 7 Days"
        elif period == '30days':
            start_date = (now - timedelta(days=30)).normalize()
            df_filtered = df_analyzable[df_analyzable['Date_dt'] >= start_date]
            period_text = "Last 30 Days"
        elif period == 'all':
            df_filtered = df_analyzable.copy()
            period_text = "All Time"
        else:
            logging.warning(f"Unknown period '{period}' received. Defaulting to 'all'.")
            df_filtered = df_analyzable.copy()
            period_text = "All Time"
        stats_period_title.config(text=f"Statistics for: {period_text}")
        logging.info(f"Filtered history for '{period_text}'. Original analyzable rows: {len(df_analyzable)}, Filtered rows: {len(df_filtered)}")
    except Exception as e:
        logging.error(f"Error filtering DataFrame for period '{period}': {e}", exc_info=True)
        for key, label in stats_labels.items():
             label.config(text=f"{label.cget('text').split(':')[0]}: Error - Filter", foreground=LOSS_COLOR)
        stats_period_title.config(text=f"Statistics Error: Failed to Filter ({period})")
        return
    # --- End Date Parsing and Filtering ---

    # --- Helper function to update labels with text and color ---
    def update_label(key, base_text, value=None, is_currency=False, is_percent=False, show_sign=False):
        if key not in stats_labels:
            logging.warning(f"Label key '{key}' not found in stats_labels dictionary.")
            return

        label = stats_labels[key]
        display_text = f"{base_text}: "
        fg_color = TEXT_COLOR # Default

        if value is None or value == '(todo)' or (isinstance(value, (float, np.floating)) and pd.isna(value)):
            display_text += "N/A" if value != '(todo)' else "(todo)"
        elif isinstance(value, str) and value == "Error":
             display_text += "Error"
             fg_color = LOSS_COLOR
        elif isinstance(value, (int, float, np.number)): # Check if it's a number
            # Determine color based on value for specific keys
            if show_sign and value > 0:
                fg_color = WIN_COLOR
            elif value < 0:
                fg_color = LOSS_COLOR

            # Format value
            if is_currency:
                display_text += f"€{value:+.2f}" if show_sign else f"€{value:.2f}"
            elif is_percent:
                display_text += f"{value:.1f}%"
            elif isinstance(value, float):
                 display_text += f"{value:.2f}" # Default float format
            else:
                 display_text += str(value) # Default int/other format
        else:
            display_text += str(value) # Non-numeric, non-error value

        label.config(text=display_text, foreground=fg_color)
    # --- End Helper Function ---

    # --- Calculate Statistics from df_filtered ---
    try:
        if df_filtered.empty:
            logging.warning(f"No betting history found for the selected period: {period_text}")
            # Set all labels to N/A using the helper
            update_label('total_bets', "Total Bets", 0)
            update_label('completed', "Completed", 0)
            update_label('pending', "Pending", 0)
            update_label('wins', "Wins", 0)
            update_label('losses', "Losses", 0)
            update_label('win_rate', "Win Rate", None)
            update_label('total_stake', "Total Stake", 0, is_currency=True)
            update_label('total_profit', "Total Profit", 0, is_currency=True, show_sign=True)
            update_label('roi', "ROI", None)
            update_label('avg_stake', "Avg Stake", None)
            update_label('avg_odds', "Avg Odds (Placed)", None)
            update_label('biggest_win', "Biggest Win", None)
            update_label('biggest_loss', "Biggest Loss", None)
            update_label('longest_win_streak', "Longest Win Streak", 0)
            update_label('longest_loss_streak', "Longest Loss Streak", 0)
            update_label('profit_std_dev', "Profit Std Dev", None)
            # (Top Leagues/Games by League updates later)
            logging.info("--- update_stats_display finished (no data for period) ---")
            return

        # Ensure numeric types
        df_filtered['Stake'] = pd.to_numeric(df_filtered['Stake'], errors='coerce').fillna(0)
        df_filtered['Odds'] = pd.to_numeric(df_filtered['Odds'], errors='coerce').fillna(0)
        df_filtered['Payout'] = pd.to_numeric(df_filtered['Payout'], errors='coerce').fillna(0)

        # --- Calculations ---
        total_bets_count = len(df_filtered)
        pending_bets = df_filtered[df_filtered['Result'] == 'Pending']
        finished_bets = df_filtered[df_filtered['Result'].isin(['Win', 'Loss'])]
        won_bets = finished_bets[finished_bets['Result'] == 'Win']
        lost_bets = finished_bets[finished_bets['Result'] == 'Loss']
        pending_count = len(pending_bets)
        completed_count = len(finished_bets)
        wins_count = len(won_bets)
        losses_count = len(lost_bets)

        win_rate = (wins_count / completed_count * 100) if completed_count > 0 else None

        total_stake = finished_bets['Stake'].sum()

        def calculate_profit(row):
            if row['Result'] == 'Win': return row['Payout'] - row['Stake']
            elif row['Result'] == 'Loss': return -row['Stake']
            else: return 0
        # Use .loc to avoid SettingWithCopyWarning if possible
        finished_bets_copy = finished_bets.copy()
        finished_bets_copy['Profit'] = finished_bets_copy.apply(calculate_profit, axis=1)
        total_profit = finished_bets_copy['Profit'].sum()
        
        roi = (total_profit / total_stake * 100) if total_stake > 0 else None
        avg_stake = total_stake / completed_count if completed_count > 0 else None
        avg_odds = finished_bets['Odds'].mean() if completed_count > 0 else None
        biggest_win = finished_bets_copy.loc[finished_bets_copy['Profit'] > 0, 'Profit'].max() if wins_count > 0 else None
        biggest_loss = finished_bets_copy.loc[finished_bets_copy['Profit'] < 0, 'Profit'].min() if losses_count > 0 else None
        profit_std_dev = finished_bets_copy['Profit'].std() if completed_count > 1 else None

        # --- Streaks Calculation ---
        longest_win_streak = 0
        current_win_streak = 0
        longest_loss_streak = 0
        current_loss_streak = 0
        if completed_count > 0:
            # Sort finished bets by date to calculate streaks correctly
            # Use the Date_dt column we created earlier for sorting
            sorted_finished_bets = finished_bets_copy.sort_values(by='Date_dt')
            for result in sorted_finished_bets['Result']:
                if result == 'Win':
                    current_win_streak += 1
                    current_loss_streak = 0 # Reset loss streak
                    if current_win_streak > longest_win_streak:
                        longest_win_streak = current_win_streak
                elif result == 'Loss':
                    current_loss_streak += 1
                    current_win_streak = 0 # Reset win streak
                    if current_loss_streak > longest_loss_streak:
                        longest_loss_streak = current_loss_streak
                # Ignore other results if any (shouldn't happen with current filtering)
                
        # --- End Streaks Calculation ---

        # --- Update Labels using Helper ---
        update_label('total_bets', "Total Bets", total_bets_count)
        update_label('completed', "Completed", completed_count)
        update_label('pending', "Pending", pending_count)
        update_label('wins', "Wins", wins_count)
        update_label('losses', "Losses", losses_count)
        update_label('win_rate', "Win Rate", win_rate, is_percent=True)
        update_label('total_stake', "Total Stake", total_stake, is_currency=True)
        update_label('total_profit', "Total Profit", total_profit, is_currency=True, show_sign=True)
        update_label('roi', "ROI", roi, is_percent=True)
        update_label('avg_stake', "Avg Stake", avg_stake, is_currency=True)
        update_label('avg_odds', "Avg Odds (Placed)", avg_odds)
        update_label('biggest_win', "Biggest Win", biggest_win, is_currency=True, show_sign=True)
        update_label('biggest_loss', "Biggest Loss", biggest_loss, is_currency=True)
        # Update streak labels with calculated values
        update_label('longest_win_streak', "Longest Win Streak", longest_win_streak)
        update_label('longest_loss_streak', "Longest Loss Streak", longest_loss_streak)
        update_label('profit_std_dev', "Profit Std Dev", profit_std_dev, is_currency=True)

        # --- Top 3 Profitable Leagues --- 
        if completed_count > 0 and 'League' in finished_bets_copy.columns:
            try:
                league_profit = finished_bets_copy.groupby('League')['Profit'].sum()
                league_profit = league_profit[(league_profit > 0) & (league_profit.index.str.strip() != "") & (league_profit.index.str.lower() != "unknown league")] # Only positive profit
                top_leagues = league_profit.nlargest(3)
                
                # Update the dedicated labels
                for i in range(3):
                    if i < len(top_leagues):
                        league = top_leagues.index[i]
                        profit = top_leagues.iloc[i]
                        # Apply green color for profit
                        top_league_labels[i].config(text=f"{i+1}. {league}: €{profit:+.2f}", foreground=WIN_COLOR)
                    else:
                        top_league_labels[i].config(text="") # Clear unused labels
                        
                if top_leagues.empty:
                     top_league_labels[0].config(text="No profitable leagues found.", foreground=TEXT_COLOR) # Use first label for message

            except Exception as league_ex:
                logging.error(f"Error calculating top leagues: {league_ex}", exc_info=True)
                for i in range(3):
                     top_league_labels[i].config(text="Error calculating.", foreground=LOSS_COLOR) 
                     
        elif 'League' not in finished_bets_copy.columns:
             for i in range(3):
                  top_league_labels[i].config(text="Error: League column missing.", foreground=LOSS_COLOR)
        else: # No completed bets
             for i in range(3):
                  top_league_labels[i].config(text="", foreground=TEXT_COLOR) # Clear labels if no bets
        # --- End Top 3 Leagues ---

        # --- Update "By League" Tab --- 
        # Clear previous entries
        for item in league_stats_tree.get_children():
            league_stats_tree.delete(item)
            
        if completed_count > 0 and 'League' in df_filtered.columns: # Check df_filtered now
            try:
                # Create a working copy to add Profit column safely
                df_working_league = df_filtered.copy()
                # Ensure Profit calculation uses the correct function defined earlier
                df_working_league['Profit'] = df_working_league.apply(calculate_profit, axis=1)

                # Group by league and calculate aggregates
                # Fill NaN league names before grouping to avoid issues
                df_working_league['League'].fillna("Unknown League", inplace=True)
                league_groups = df_working_league.groupby('League')

                # Aggregate stats using named aggregation for clarity
                league_stats = league_groups.agg(
                    total_bets = pd.NamedAgg(column='Match', aggfunc='size'),
                    wins = pd.NamedAgg(column='Result', aggfunc=lambda x: (x == 'Win').sum()),
                    losses = pd.NamedAgg(column='Result', aggfunc=lambda x: (x == 'Loss').sum()),
                    # Calculate stake only on finished bets within the group
                    total_stake_finished = pd.NamedAgg(column='Stake', aggfunc=lambda x: x[df_working_league.loc[x.index, 'Result'].isin(['Win', 'Loss'])].sum()),
                    total_profit = pd.NamedAgg(column='Profit', aggfunc='sum')
                ).reset_index()

                # Calculate derived stats
                league_stats['Completed'] = league_stats['wins'] + league_stats['losses']
                # Handle division by zero for Win Rate and ROI
                league_stats['Win Rate'] = (league_stats['wins'] / league_stats['Completed'] * 100).where(league_stats['Completed'] > 0, 0)
                league_stats['ROI'] = (league_stats['total_profit'] / league_stats['total_stake_finished'] * 100).where(league_stats['total_stake_finished'] > 0, 0)

                # Sort by Profit by default (optional)
                league_stats = league_stats.sort_values(by="total_profit", ascending=False)

                # Populate Treeview (By League)
                if league_stats.empty:
                    league_stats_tree.insert("", "end", values=("No league data to display...", "", "", "", "", "", "", "", ""))
                else:
                    for index, row in league_stats.iterrows():
                        profit_val = row.get('total_profit', 0)
                        
                        # --- Determine Profit Symbol/Color --- 
                        profit_symbol = ""
                        # Use Unicode characters for symbols; ensure font supports them
                        if profit_val > 0:
                             # Green Circle (or choose another symbol like ▲)
                             profit_symbol = f'\U0001F7E2 ' # Green Circle Unicode
                        elif profit_val < 0:
                             # Red Circle (or choose another symbol like ▼)
                             profit_symbol = f'\U0001F534 ' # Red Circle Unicode
                        # Add a space if no symbol for alignment, or use a neutral symbol like a gray dot
                        else:
                            profit_symbol = "  " # Space for alignment
                            
                        # Format values for display
                        league_name = row.get('League', 'N/A')
                        if pd.isna(league_name): league_name = "Unknown League"
                        else: league_name = str(league_name)
                        bets_count = row.get('total_bets', 0)
                        completed_val = row.get('Completed', 0)
                        wins_val = row.get('wins', 0)
                        losses_val = row.get('losses', 0)
                        win_rate_str = f"{row.get('Win Rate', 0):.1f}%"
                        stake_str = f"€{row.get('total_stake_finished', 0):.2f}"
                        # Prepend symbol to the formatted profit string
                        profit_str = f"{profit_symbol}€{profit_val:+.2f}"
                        roi_str = f"{row.get('ROI', 0):.1f}%"
                        
                        # Insert row without tags
                        league_stats_tree.insert("", "end", values=(
                            league_name, bets_count, completed_val, wins_val, losses_val,
                            win_rate_str, stake_str, profit_str, roi_str
                        )) # No tags
            except Exception as league_tab_ex:
                 # This except block needs to be correctly aligned with the try block
                 logging.error(f"Error populating 'By League' tab: {league_tab_ex}", exc_info=True)
                 league_stats_tree.insert("", "end", values=("Error loading league data...", "", "", "", "", "", "", "", ""))
                 
        elif 'League' not in df_filtered.columns:
             # This elif and the next ones also need correct alignment
             league_stats_tree.insert("", "end", values=("League column missing...", "", "", "", "", "", "", "", ""))
        elif completed_count == 0 and not df_filtered.empty :
             league_stats_tree.insert("", "end", values=("No completed bets in period...", "", "", "", "", "", "", "", ""))
        elif df_filtered.empty:
             league_stats_tree.insert("", "end", values=("No bets in selected period...", "", "", "", "", "", "", "", ""))
        # --- End "By League" Tab Update ---

        # --- Update "By Odds" Tab ---
        # Clear previous entries
        for item in odds_stats_tree.get_children():
            odds_stats_tree.delete(item)
            
        if completed_count > 0:
            try:
                # Define odds bins and labels
                odds_bins = [0, 1.5, 2.0, 3.0, 5.0, float('inf')] # Bins: 0-1.49, 1.5-1.99, 2.0-2.99, 3.0-4.99, 5.0+
                odds_labels = ['< 1.50', '1.50 - 1.99', '2.00 - 2.99', '3.00 - 4.99', '5.00+']

                # Create a working copy with Profit (already done for league stats)
                df_working_odds = df_filtered.copy()
                # Ensure the Profit column exists (it should from league calculation part)
                if 'Profit' not in df_working_odds.columns:
                     df_working_odds['Profit'] = df_working_odds.apply(calculate_profit, axis=1)

                # Assign odds range category to each bet
                # Ensure Odds column is numeric
                df_working_odds['Odds'] = pd.to_numeric(df_working_odds['Odds'], errors='coerce')
                df_working_odds.dropna(subset=['Odds'], inplace=True) # Drop rows where Odds couldn't be parsed
                
                df_working_odds['Odds Range'] = pd.cut(df_working_odds['Odds'], bins=odds_bins, labels=odds_labels, right=False, include_lowest=True)

                # Group by the new Odds Range category
                odds_groups = df_working_odds.groupby('Odds Range', observed=False) # observed=False includes all categories even if empty

                # Aggregate stats (similar to leagues)
                odds_stats = odds_groups.agg(
                    total_bets = pd.NamedAgg(column='Match', aggfunc='size'),
                    wins = pd.NamedAgg(column='Result', aggfunc=lambda x: (x == 'Win').sum()),
                    losses = pd.NamedAgg(column='Result', aggfunc=lambda x: (x == 'Loss').sum()),
                    total_stake_finished = pd.NamedAgg(column='Stake', aggfunc=lambda x: x[df_working_odds.loc[x.index, 'Result'].isin(['Win', 'Loss'])].sum()),
                    total_profit = pd.NamedAgg(column='Profit', aggfunc='sum')
                ).reset_index()

                # Calculate derived stats
                odds_stats['Completed'] = odds_stats['wins'] + odds_stats['losses']
                odds_stats['Win Rate'] = (odds_stats['wins'] / odds_stats['Completed'] * 100).where(odds_stats['Completed'] > 0, 0)
                odds_stats['ROI'] = (odds_stats['total_profit'] / odds_stats['total_stake_finished'] * 100).where(odds_stats['total_stake_finished'] > 0, 0)

                # Populate Treeview (By Odds)
                if odds_stats.empty:
                     odds_stats_tree.insert("", "end", values=("No odds data to display...", "", "", "", "", "", "", "", ""))
                else:
                    for index, row in odds_stats.iterrows():
                        profit_val = row.get('total_profit', 0)
                        
                        # --- Determine Profit Symbol/Color --- 
                        profit_symbol = ""
                        if profit_val > 0:
                            profit_symbol = f'\U0001F7E2 ' # Green Circle
                        elif profit_val < 0:
                            profit_symbol = f'\U0001F534 ' # Red Circle
                        else:
                            profit_symbol = "  " # Space for alignment
                        
                        # Format values
                        range_name = row.get('Odds Range', 'N/A')
                        bets_count = row.get('total_bets', 0)
                        completed_val = row.get('Completed', 0)
                        wins_val = row.get('wins', 0)
                        losses_val = row.get('losses', 0)
                        win_rate_str = f"{row.get('Win Rate', 0):.1f}%"
                        stake_str = f"€{row.get('total_stake_finished', 0):.2f}"
                        # Prepend symbol to the formatted profit string
                        profit_str = f"{profit_symbol}€{profit_val:+.2f}"
                        roi_str = f"{row.get('ROI', 0):.1f}%"
                        
                        # Insert row without tags
                        odds_stats_tree.insert("", "end", values=(
                            range_name, bets_count, completed_val, wins_val, losses_val,
                            win_rate_str, stake_str, profit_str, roi_str
                        )) # No tags
            except Exception as odds_tab_ex:
                 # Ensure correct alignment for except block
                 logging.error(f"Error populating 'By Odds' tab: {odds_tab_ex}", exc_info=True)
                 odds_stats_tree.insert("", "end", values=("Error loading odds data...", "", "", "", "", "", "", "", ""))
                 
        elif completed_count == 0: # Handle case with no completed bets for any odds calc
             # Ensure correct alignment for elif blocks
             odds_stats_tree.insert("", "end", values=("No completed bets in period...", "", "", "", "", "", "", "", ""))
        elif df_filtered.empty:
             odds_stats_tree.insert("", "end", values=("No bets in selected period...", "", "", "", "", "", "", "", ""))
        # --- End "By Odds" Tab Update ---

        logging.info(f"Stats updated for period '{period_text}'")

    except Exception as e:
        logging.error(f"Error calculating specific stats for period '{period_text}': {e}", exc_info=True)
        for key, label in stats_labels.items():
             update_label(key, label.cget('text').split(':')[0], value="Error") # Use helper for error state
    finally:
        logging.info("--- update_stats_display finished ---")

def show_stats_page():
    """Hides the main frames and shows the stats frame."""
    logging.info("--- show_stats_page entered ---") # Log entry
    # logging.debug("Switching to Stats Page")
    # Hide main frames
    br_frame.grid_remove()
    feedback_frame.grid_remove()
    league_filter_frame.grid_remove()
    history_frame.grid_remove()
    # Show stats frame
    stats_frame.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=20, pady=10) # Span rows 0-3 in the main grid
    # Update the stats display when showing the page
    update_stats_display()
    # --- DEBUGGING START ---
    # print(f"DEBUG: stats_total_bets_label text = '{stats_total_bets_label.cget('text')}'") # Removed debug
    # print(f"DEBUG: stats_win_rate_label text = '{stats_win_rate_label.cget('text')}'") # Removed debug
    # --- DEBUGGING END ---

# --- End of Stats Page Functionality ---

# --- GUI Setup ---
root = TkinterDnD.Tk()
logging.info("TkinterDnD root created.")
root.title("Betting Calculator")
root.geometry("1600x900") # <-- Increased initial size to 1600x900

# --- Set App Icon ---
try:
    # Construct absolute path relative to the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(script_dir, "app_icon.png")
    
    # Use iconphoto for cross-platform compatibility with PNG
    icon_image = tk.PhotoImage(file=icon_path) 
    root.iconphoto(True, icon_image) # True makes it apply to toplevel windows too
    logging.info(f"Attempting to set application icon from: {icon_path}")
except tk.TclError as e:
    logging.warning(f"Could not load or set application icon ({icon_path}): {e}. Make sure the file exists and is a valid PNG.")
except FileNotFoundError:
     logging.warning(f"Could not find application icon file at: {icon_path}")
except Exception as e:
    logging.error(f"Unexpected error setting application icon: {e}", exc_info=True)
# --- End App Icon ---

root.configure(bg="#2D2D2D") # Use defined background color

# Variables
bankroll = load_bankroll()
history = load_history()

# Styles - Apply a more consistent dark theme
style = ttk.Style()
try:
    style.theme_use('clam') 
    logging.info("Using 'clam' theme.")
except tk.TclError:
    logging.warning("Clam theme not available, using default.")
    style.theme_use('default')

# General widget styling
style.configure(".", 
                background="#2D2D2D", 
                foreground="#EAEAEA", # Light gray text
                font=("Segoe UI", 10)) 
style.configure("TFrame", background="#2D2D2D") # Ensure frames default to main BG

# --- Button Style --- 
style.configure("TButton", 
                font=("Segoe UI", 10, "normal"), # Changed from bold
                padding=(10, 6), # Adjusted padding (horizontal, vertical)
                background="#555555", # Medium gray button
                foreground="#EAEAEA",
                borderwidth=0, # Flat look
                relief="flat") 
style.map("TButton",
          background=[('active', "#666666")], # Slightly lighter on hover/press
          foreground=[('active', "#EAEAEA")]) 

# --- Label Styles --- 
# Default Label on main background
style.configure("TLabel", 
                font=("Segoe UI", 11), 
                background="#2D2D2D", 
                foreground="#EAEAEA")
# Label inside a card
style.configure("Card.TLabel", 
                font=("Segoe UI", 10), 
                background="#3C3C3C", # Slightly lighter card background
                foreground="#A0A0A0") # Muted text inside cards
# Card Title Label
style.configure("CardTitle.TLabel", 
                font=("Segoe UI", 13, "bold"), 
                background="#3C3C3C", # Slightly lighter card background
                foreground=TEXT_COLOR) # <-- Changed from #5AC8FA to TEXT_COLOR
# Feedback Label
style.configure("Feedback.TLabel", # Feedback specifically
                font=("Segoe UI", 10, "italic"), 
                background="#3C3C3C", # Match card background
                foreground="#A0A0A0", 
                padding=(5, 5))
# Bankroll Amount Label
style.configure("Bankroll.TLabel", 
                font=("Segoe UI", 18, "bold"), 
                background="#3C3C3C", # Slightly lighter card background
                foreground="#EAEAEA") # White text

# --- Combobox Style ---
style.configure("TCombobox", 
                font=("Segoe UI", 10),
                padding=5,
                background="#555555",
                foreground="#EAEAEA",
                fieldbackground="#555555",
                arrowcolor="#A0A0A0",
                selectbackground="#666666",
                selectforeground="#EAEAEA",
                borderwidth=0,
                relief="flat")
# Ensure dropdown list uses similar colors
root.option_add('*TCombobox*Listbox.background', "#555555")
root.option_add('*TCombobox*Listbox.foreground', "#EAEAEA")
root.option_add('*TCombobox*Listbox.selectBackground', "#007AFF")
root.option_add('*TCombobox*Listbox.selectForeground', "#EAEAEA")

style.map('TCombobox', fieldbackground=[('readonly', "#555555")])
style.map('TCombobox', selectbackground=[('readonly', "#666666")])
style.map('TCombobox', selectforeground=[('readonly', "#EAEAEA")])

# --- Entry Style --- 
style.configure("TEntry", 
                font=("Segoe UI", 10),
                foreground="#EAEAEA",          # White text
                fieldbackground="#444444",    # Dark gray background
                insertcolor='#EAEAEA',        # White cursor
                borderwidth=0,
                relief="flat")
style.map("TEntry",
          foreground=[('disabled', '#888888')], # Grayer text when disabled
          fieldbackground=[('disabled', '#555555')])

# --- Treeview Style --- 
style.configure("Treeview",
                font=("Segoe UI", 10),
                background="#3C3C3C", 
                foreground="#EAEAEA",
                fieldbackground="#3C3C3C",
                rowheight=28, # Slightly taller rows
                borderwidth=0,
                relief="flat") 

style.configure("Treeview.Heading",
                font=("Segoe UI", 10, "bold"),
                background="#555555", # Match card background for seamless look
                foreground="#5AC8FA", # Heading color
                padding=(10, 5),
                relief="flat",
                borderwidth=0) 
style.map("Treeview.Heading",
          background=[('active', "#666666")]) # Keep heading background same on click

# Selected item style
style.map('Treeview',
          background=[('selected', "#007AFF")], # Use accent color for selection
          foreground=[('selected', "#EAEAEA")])

# Tag configurations (Foreground colors for Result column)
# Note: We use foreground for the Result text itself based on the tag.
style.configure("win.Treeview", foreground="#34C759") 
style.configure("loss.Treeview", foreground="#FF3B30") 
style.configure("pending.Treeview", foreground="#A0A0A0") # Muted for pending

# Stripe tag - slightly different background for alternate rows
# Using a slightly darker shade of the card background for striping
#STRIPE_COLOR = "#353535" # Adjusted stripe color slightly
#style.configure("striped.Treeview", background=STRIPE_COLOR)
# Ensure selected striped rows still use the main selection color
#style.map('Treeview',
#         background=[('selected', 'striped', "#007AFF"), ('selected', "#007AFF")],
#         foreground=[('selected', "#EAEAEA")])

# --- Card Frame Style (using tk.Frame as base) ---
# We will apply colors directly when creating frames now, instead of this dict
# card_style = {
#     "bg": CARD_BG_COLOR,
#     "bd": 0, # No border
#     "relief": "flat" 
# }

# --- Separator Style ---
style.configure("TSeparator", background="#505050")


# --- Main Frames Setup ---
# Create frames directly with the card background color
br_frame = tk.Frame(root, bg="#3C3C3C")
feedback_frame = tk.Frame(root, bg="#3C3C3C") # Frame for feedback label
league_filter_frame = tk.Frame(root, bg="#3C3C3C")
history_frame = tk.Frame(root, bg="#3C3C3C")
stats_frame = tk.Frame(root, bg="#2D2D2D") # Stats main container still uses root bg

# --- Stats Frame Content - REBUILD with TABS ---

# Configure Grid for stats_frame (main container for back button, filters, notebook)
stats_frame.grid_columnconfigure(0, weight=1)
stats_frame.grid_rowconfigure(0, weight=0) # Back button row
stats_frame.grid_rowconfigure(1, weight=0) # Time filter row
stats_frame.grid_rowconfigure(2, weight=1) # Notebook row (expands)

# --- Back Button --- (Remains at top left of stats_frame)
# Need to ensure this button is created before the notebook if it needs to be visually above it
back_button_stats = ttk.Button(stats_frame, text="← Back to Bets", command=show_main_page, style="TButton")
back_button_stats.grid(row=0, column=0, padx=20, pady=(10,5), sticky="nw") # Adjust padding

# --- Time Period Filters --- (Remain below back button)
time_filter_frame = tk.Frame(stats_frame, bg="#2D2D2D")
time_filter_frame.grid(row=1, column=0, pady=(0, 10), padx=20, sticky="ew")
ttk.Label(time_filter_frame, text="Select Period:", background="#2D2D2D", foreground="#EAEAEA").pack(side=tk.LEFT, padx=(0, 10))
ttk.Button(time_filter_frame, text="Today", command=lambda: update_stats_display(period='today'), style="TButton").pack(side=tk.LEFT, padx=5)
ttk.Button(time_filter_frame, text="Last 7 Days", command=lambda: update_stats_display(period='7days'), style="TButton").pack(side=tk.LEFT, padx=5)
ttk.Button(time_filter_frame, text="Last 30 Days", command=lambda: update_stats_display(period='30days'), style="TButton").pack(side=tk.LEFT, padx=5)
ttk.Button(time_filter_frame, text="All Time", command=lambda: update_stats_display(period='all'), style="TButton").pack(side=tk.LEFT, padx=5)

# --- Notebook for Tabs ---
# Define style specifically for the notebook widget
notebook_style = ttk.Style()
notebook_style.configure('Stats.TNotebook', tabposition='nw') # Place tabs at top-left
notebook_style.configure('Stats.TNotebook', background="#2D2D2D", borderwidth=0)
notebook_style.configure('Stats.TNotebook.Tab',
                         background="#555555",
                         foreground="#EAEAEA",
                         padding=[10, 5],
                         font=("Segoe UI", 10),
                         borderwidth=0)
notebook_style.map('Stats.TNotebook.Tab',
                   background=[('selected', "#3C3C3C"), ('active', "#666666")],
                   foreground=[('selected', TEXT_COLOR), ('active', TEXT_COLOR)])

stats_notebook = ttk.Notebook(stats_frame, style='Stats.TNotebook')
stats_notebook.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 10))

# --- Create Frames for Tabs --- (Use main background color)
overview_tab = tk.Frame(stats_notebook, bg="#2D2D2D")
by_league_tab = tk.Frame(stats_notebook, bg="#2D2D2D")
by_odds_tab = tk.Frame(stats_notebook, bg="#2D2D2D")

# Add tabs to notebook
stats_notebook.add(overview_tab, text='Overview')
stats_notebook.add(by_league_tab, text='By League')
stats_notebook.add(by_odds_tab, text='By Odds')

# --- Configure Grid within Overview Tab --- 
overview_tab.grid_columnconfigure(0, weight=1) # Spacer
overview_tab.grid_columnconfigure(1, weight=0) # Main stats col
overview_tab.grid_columnconfigure(2, weight=1) # Spacer
overview_tab.grid_columnconfigure(3, weight=0) # Top leagues col
overview_tab.grid_columnconfigure(4, weight=1) # Spacer
overview_tab.grid_rowconfigure(0, weight=0) # Title row
overview_tab.grid_rowconfigure(1, weight=1) # Content row

# --- Move Existing Content to Overview Tab --- 
# Stats Title (now belongs conceptually to the tab, but might be redundant with tab name)
stats_period_title = ttk.Label(overview_tab, text="Statistics for: All Time", style="CardTitle.TLabel", foreground="#5AC8FA", background="#2D2D2D")
stats_period_title.grid(row=0, column=1, columnspan=3, pady=(15, 10), sticky="w") # Added top padding

# Main Stats Area (inside overview_tab)
stats_content_frame = tk.Frame(overview_tab, bg="#2D2D2D")
stats_content_frame.grid(row=1, column=1, sticky="nw", padx=(0, 20))

# Labels Dictionary (Ensure it's defined before use)
stats_labels = {} # Initialize dictionary to store labels

def create_stat_label(parent, key, text):
    # This function now creates labels inside stats_content_frame
    label = ttk.Label(parent, text=f"{text}: -", style="TLabel", background="#2D2D2D")
    label.pack(anchor="w", pady=1)
    stats_labels[key] = label # Store label using its key

# Create the labels within stats_content_frame
create_stat_label(stats_content_frame, "total_bets", "Total Bets")
create_stat_label(stats_content_frame, "completed", "Completed")
create_stat_label(stats_content_frame, "pending", "Pending")
create_stat_label(stats_content_frame, "wins", "Wins")
create_stat_label(stats_content_frame, "losses", "Losses")
create_stat_label(stats_content_frame, "win_rate", "Win Rate")
# Add visual separators using ttk.Separator
ttk.Separator(stats_content_frame, orient='horizontal').pack(fill='x', pady=5)
create_stat_label(stats_content_frame, "total_stake", "Total Stake")
create_stat_label(stats_content_frame, "total_profit", "Total Profit")
create_stat_label(stats_content_frame, "roi", "ROI")
ttk.Separator(stats_content_frame, orient='horizontal').pack(fill='x', pady=5)
create_stat_label(stats_content_frame, "avg_stake", "Avg Stake")
create_stat_label(stats_content_frame, "avg_odds", "Avg Odds (Placed)")
create_stat_label(stats_content_frame, "biggest_win", "Biggest Win")
create_stat_label(stats_content_frame, "biggest_loss", "Biggest Loss")
ttk.Separator(stats_content_frame, orient='horizontal').pack(fill='x', pady=5)
create_stat_label(stats_content_frame, "longest_win_streak", "Longest Win Streak")
create_stat_label(stats_content_frame, "longest_loss_streak", "Longest Loss Streak")
create_stat_label(stats_content_frame, "profit_std_dev", "Profit Std Dev")

# Top Leagues Area (inside overview_tab)
top_leagues_frame = tk.Frame(overview_tab, bg="#2D2D2D")
top_leagues_frame.grid(row=1, column=3, sticky="nw", padx=(20, 0))

top_leagues_title = ttk.Label(top_leagues_frame, text="Top 3 Profitable Leagues", style="CardTitle.TLabel", foreground="#5AC8FA", background="#2D2D2D")
top_leagues_title.pack(anchor="w", pady=(0, 10))

# --- Create 3 dedicated labels for top leagues --- 
top_league_labels = []
for i in range(3):
    # Create label with default text and styling
    label = ttk.Label(top_leagues_frame, 
                      text="", 
                      justify=tk.LEFT, 
                      style="TLabel", 
                      background="#2D2D2D", 
                      foreground=TEXT_COLOR) # Start with default text color
    label.pack(anchor="w", pady=1) # Pack below title
    top_league_labels.append(label)
# --- End dedicated labels ---

# --- Placeholder Content for Other Tabs ---
# Remove placeholder label
# ttk.Label(by_league_tab, text="League statistics breakdown coming soon...", style="TLabel").pack(padx=20, pady=20)

# --- "By League" Tab Content ---
by_league_tab.grid_rowconfigure(0, weight=1)
by_league_tab.grid_columnconfigure(0, weight=1)

league_stats_columns = ("League", "Bets", "Completed", "Wins", "Losses", "Win Rate", "Stake", "Profit", "ROI")
league_stats_tree = ttk.Treeview(by_league_tab, 
                                 columns=league_stats_columns, 
                                 show="headings", 
                                 style="Treeview") # Reuse main Treeview style

# Configure Headings
league_stats_tree.heading("League", text="League")
league_stats_tree.heading("Bets", text="Bets")
league_stats_tree.heading("Completed", text="Completed")
league_stats_tree.heading("Wins", text="Wins")
league_stats_tree.heading("Losses", text="Losses")
league_stats_tree.heading("Win Rate", text="Win Rate (%)")
league_stats_tree.heading("Stake", text="Stake (€)")
league_stats_tree.heading("Profit", text="Profit (€)")
league_stats_tree.heading("ROI", text="ROI (%)")

# Configure Column Widths and Anchors
league_stats_tree.column("League", width=200, anchor="w")
league_stats_tree.column("Bets", width=60, anchor="center")
league_stats_tree.column("Completed", width=80, anchor="center")
league_stats_tree.column("Wins", width=60, anchor="center")
league_stats_tree.column("Losses", width=60, anchor="center")
league_stats_tree.column("Win Rate", width=90, anchor="e")
league_stats_tree.column("Stake", width=100, anchor="e")
league_stats_tree.column("Profit", width=100, anchor="e")
league_stats_tree.column("ROI", width=90, anchor="e")

# Scrollbar
league_stats_scrollbar = ttk.Scrollbar(by_league_tab, orient="vertical", command=league_stats_tree.yview)
league_stats_tree.configure(yscrollcommand=league_stats_scrollbar.set)

# Grid layout
league_stats_tree.grid(row=0, column=0, sticky='nsew', padx=(10, 0), pady=10)
league_stats_scrollbar.grid(row=0, column=1, sticky='ns', padx=(0, 10), pady=10)

# Add tags for coloring Profit/ROI (similar to main history table)
league_stats_tree.tag_configure('profit_win', foreground=WIN_COLOR)
league_stats_tree.tag_configure('profit_loss', foreground=LOSS_COLOR)
# --- End "By League" Tab Content ---

# --- "By Odds" Tab Content ---
# Remove placeholder label
# ttk.Label(by_odds_tab, text="Odds range statistics breakdown coming soon...", style="TLabel").pack(padx=20, pady=20)

by_odds_tab.grid_rowconfigure(0, weight=1)
by_odds_tab.grid_columnconfigure(0, weight=1)

odds_stats_columns = ("Range", "Bets", "Completed", "Wins", "Losses", "Win Rate", "Stake", "Profit", "ROI")
odds_stats_tree = ttk.Treeview(by_odds_tab,
                               columns=odds_stats_columns,
                               show="headings",
                               style="Treeview")

# Configure Headings
odds_stats_tree.heading("Range", text="Odds Range")
odds_stats_tree.heading("Bets", text="Bets")
odds_stats_tree.heading("Completed", text="Completed")
odds_stats_tree.heading("Wins", text="Wins")
odds_stats_tree.heading("Losses", text="Losses")
odds_stats_tree.heading("Win Rate", text="Win Rate (%)")
odds_stats_tree.heading("Stake", text="Stake (€)")
odds_stats_tree.heading("Profit", text="Profit (€)")
odds_stats_tree.heading("ROI", text="ROI (%)")

# Configure Column Widths and Anchors
odds_stats_tree.column("Range", width=120, anchor="w") # Range might be shorter text
odds_stats_tree.column("Bets", width=60, anchor="center")
odds_stats_tree.column("Completed", width=80, anchor="center")
odds_stats_tree.column("Wins", width=60, anchor="center")
odds_stats_tree.column("Losses", width=60, anchor="center")
odds_stats_tree.column("Win Rate", width=90, anchor="e")
odds_stats_tree.column("Stake", width=100, anchor="e")
odds_stats_tree.column("Profit", width=100, anchor="e")
odds_stats_tree.column("ROI", width=90, anchor="e")

# Scrollbar
odds_stats_scrollbar = ttk.Scrollbar(by_odds_tab, orient="vertical", command=odds_stats_tree.yview)
odds_stats_tree.configure(yscrollcommand=odds_stats_scrollbar.set)

# Grid layout
odds_stats_tree.grid(row=0, column=0, sticky='nsew', padx=(10, 0), pady=10)
odds_stats_scrollbar.grid(row=0, column=1, sticky='ns', padx=(0, 10), pady=10)

# Add tags for coloring Profit/ROI
odds_stats_tree.tag_configure('profit_win', foreground=WIN_COLOR)
odds_stats_tree.tag_configure('profit_loss', foreground=LOSS_COLOR)
# --- End "By Odds" Tab Content ---

# --- End Stats Frame Content REBUILD with TABS ---

# --- Bankroll Frame Content --- 
# ... existing code ...

# --- Bankroll Frame Content --- 
# Add padding inside the frame
br_frame.grid_columnconfigure(0, weight=1) # Allow button column to center/space

# Use the specific Bankroll label style
br_label = ttk.Label(br_frame, 
                    text=f"Bankroll: €{bankroll:.2f}", 
                    style="Bankroll.TLabel")
br_label.grid(row=0, column=0, columnspan=3, pady=(15, 10), padx=20) # Increased padding

# Button Frame for better layout
button_frame = tk.Frame(br_frame, bg="#3C3C3C")
button_frame.grid(row=1, column=0, columnspan=3, pady=(5, 15), padx=20)

# Use pack within the button frame for centering/spacing
button_style = {} # Remove fixed width
ttk.Button(button_frame, text="Reset Bankroll", 
           command=lambda: adjust_bankroll(200 - bankroll), 
           style="TButton",
           **button_style).pack(side=tk.LEFT, padx=10) # Add padding between buttons
ttk.Button(button_frame, text="Manual Change", 
           command=manual_bankroll_change,
           style="TButton",
           **button_style).pack(side=tk.LEFT, padx=10)
ttk.Button(button_frame, text="Show Stats",
           command=show_stats_page, # <-- Add this command back
           style="TButton",
           **button_style).pack(side=tk.LEFT, padx=10)


# --- Drop Target Info & Feedback Frame --- 
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop) 
logging.info("Root window registered as drop target.")

feedback_frame.grid_columnconfigure(0, weight=1) # Allow label to expand

upload_info_label = ttk.Label(feedback_frame, 
                              text="Drop Bet Screenshot Anywhere To Add", 
                              style="Card.TLabel", # Use card label style
                              font=("Segoe UI", 10, "italic"), # Keep italic
                              foreground="#A0A0A0")
upload_info_label.grid(row=0, column=0, pady=(10, 2), padx=10) # Adjust padding

feedback_label = ttk.Label(feedback_frame, 
                           text="App ready. Drop a screenshot.", 
                           style="Feedback.TLabel", 
                           wraplength=500, 
                           anchor="center") 
feedback_label.grid(row=1, column=0, pady=(0, 10), padx=10, sticky="ew") # Adjust padding

# --- League Filter Frame Content ---
league_filter_frame.grid_columnconfigure(1, weight=1) # Allow combobox to expand a bit

ttk.Label(league_filter_frame, 
          text="Filter by League:", 
          style="Card.TLabel", # Use standard muted card text
          font=("Segoe UI", 11) # Normal weight
         ).grid(row=0, column=0, padx=(10, 5), pady=10, sticky="w")

league_var = tk.StringVar(value="All Leagues")
# Note: Initial population happens later
league_combo = ttk.Combobox(league_filter_frame, 
                            textvariable=league_var,
                            state="readonly",
                            width=35, # Adjusted width
                            style="TCombobox") # Apply style
league_combo.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ew")

# --- History Frame Content ---
history_frame.grid_rowconfigure(1, weight=1) # Allow treeview row to expand
history_frame.grid_columnconfigure(0, weight=1) # Allow treeview col to expand

history_title = ttk.Label(history_frame, text="Bet History (Today)", style="CardTitle.TLabel")
history_title.grid(row=0, column=0, columnspan=2, pady=(10, 5), padx=10, sticky="nw") # Adjust padding

# Define columns for the Treeview (structure remains same)
columns = ("Match", "League", "Odds", "Stake", "Result", "Payout", "Date", 
           "Win ✅", "Loss ❌", "Delete 🗑️", "Index") 
           
tree = ttk.Treeview(history_frame, 
                    columns=columns,
                    show="headings",
                    displaycolumns=("Match", "League", "Odds", "Stake", "Result", "Payout", "Date", 
                                  "Win ✅", "Loss ❌", "Delete 🗑️"), 
                    height=15, 
                    style="Treeview") # Style applied here

# --- Headings and Columns setup (remains structurally same, but styles applied via config) --- 
# Configure Headings (Text only, style handles appearance)
tree.heading("Match", text="Match")
tree.heading("League", text="League")
tree.heading("Odds", text="Odds")
tree.heading("Stake", text="Stake")
tree.heading("Result", text="Result")
tree.heading("Payout", text="Payout")
tree.heading("Date", text="Date")
tree.heading("Win ✅", text="Win") # Simpler text
tree.heading("Loss ❌", text="Loss") # Simpler text
tree.heading("Delete 🗑️", text="Del") # Simpler text

# Configure Column Widths and Anchors (adjust as needed)
tree.column("Match", width=300, anchor="w")
tree.column("League", width=150, anchor="w")
tree.column("Odds", width=60, anchor="center")
tree.column("Stake", width=90, anchor="e")
tree.column("Result", width=80, anchor="center")
tree.column("Payout", width=90, anchor="e")
tree.column("Date", width=140, anchor="center")
tree.column("Win ✅", width=40, anchor="center") # Adjust width
tree.column("Loss ❌", width=40, anchor="center") # Adjust width
tree.column("Delete 🗑️", width=40, anchor="center") # Adjust width
# --- End Headings/Columns ---

# --- Scrollbar Style --- 
# Basic scrollbar, styling is limited in ttk
scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)

# Grid layout for Treeview and Scrollbar
tree.grid(row=1, column=0, sticky='nsew', padx=(10, 0), pady=(5, 10)) # Adjust padding
scrollbar.grid(row=1, column=1, sticky='ns', padx=(0, 10), pady=(5, 10))

# Bindings remain the same
tree.bind('<Button-1>', on_tree_click)
logging.info("--- <Button-1> bound to on_tree_click for history Treeview ---") 
tree.bind('<Button-3>', show_context_menu)
logging.info("--- <Button-3> bound to show_context_menu for history Treeview ---")

# ... (rest of the UI setup and mainloop) ...

# --- Layout Management --- 
# Configure grid weights for main layout 
root.grid_rowconfigure(0, weight=0) # Bankroll card
root.grid_rowconfigure(1, weight=0) # Feedback card
root.grid_rowconfigure(2, weight=0) # Filter card
root.grid_rowconfigure(3, weight=1) # History card (expands vertically)

root.grid_columnconfigure(0, weight=1) # Left spacer
root.grid_columnconfigure(1, weight=10) # <<< Main Content Column >>>
root.grid_columnconfigure(2, weight=1) # Right spacer

# Place initial frames using grid 
# Add padding between cards using pady on the grid call
br_frame.grid(row=0, column=1, sticky="ew", padx=20, pady=(10, 5))
feedback_frame.grid(row=1, column=1, sticky="ew", padx=20, pady=5)
league_filter_frame.grid(row=2, column=1, sticky="ew", padx=20, pady=5)
history_frame.grid(row=3, column=1, sticky="nsew", padx=20, pady=(5, 10))

# --- Initial Data Load ---
# Ensure initial calls happen after main layout setup
update_history_table(show_only_today=True) # Load initial history data, filtering for today
update_league_filter() # Update league filter options based on loaded history

# --- Start the main loop ---
root.mainloop()