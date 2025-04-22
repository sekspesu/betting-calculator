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
                team_name = potential_team_name
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

                    logging.info(f"[Original Format] Potential Bet: Team='{team_name}', Odds={odds:.2f}")
                    current_bet = {'team': team_name, 'odds': odds}
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
                        logging.warning(f"[Original Format] Discarding bet (Stake missing): Team='{team_name}'")
                except ValueError: 
                    logging.warning(f"[Original Format] Could not parse odds '{odds_str_raw}'. Skipping.")
        
        if not processed_bet_on_this_line:
            i += 1
    
    logging.info(f"Original Format parsing finished. Found {len(bets)} bets.")
    return bets
# <<< END HELPER FUNCTION >>>

# <<< Placeholder for Coolbet parser >>>
def _parse_format_coolbet(lines):
    logging.warning("Coolbet parsing not yet implemented.")
    # TODO: Implement parsing logic for Coolbet format here
    return []
# <<< END Placeholder >>>


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
        # Simple detection for now: Check for "Stake" label vs "Team - Team" structure
        text_lower = text.lower()
        has_stake_label = re.search(r'\bstake\b', text_lower) is not None
        # Pattern for "Team A - Team B" (allows various characters in names)
        # Corrected regex: pass re.MULTILINE as a flag argument
        team_vs_team_pattern = r'^(.+?)\s+-\s+(.+?)$'
        has_team_vs_team = re.search(team_vs_team_pattern, text, flags=re.MULTILINE) is not None 
        has_match_result_line = "match result (1x2)" in text_lower

        logging.debug(f"Format detection: has_stake_label={has_stake_label}, has_team_vs_team={has_team_vs_team}, has_match_result_line={has_match_result_line}")

        if has_stake_label and not (has_team_vs_team and has_match_result_line):
            # Likely original format if "Stake" is present and typical Coolbet structure isn't
            detected_format = "Original"
            bets = _parse_format_original(lines)
        elif has_team_vs_team and has_match_result_line:
            # Likely Coolbet format if Team-Team and Match Result line are present
            detected_format = "Coolbet"
            bets = _parse_format_coolbet(lines) # Call placeholder for now
        else:
            # Could be original format *without* a visible Stake label, or another format
            # Let's try original first as a fallback if unsure?
            logging.warning("Could not reliably detect format. Trying Original format as fallback...")
            detected_format = "Original (Fallback)"
            bets = _parse_format_original(lines)
            if not bets:
                logging.warning("Fallback Original format parsing failed. Attempting Coolbet format...")
                detected_format = "Coolbet (Fallback)"
                bets = _parse_format_coolbet(lines) # Call placeholder
        
        # --- Process results --- 
        if bets:
            logging.info(f"Successfully extracted {len(bets)} bets using {detected_format} parser.")
            feedback_text = f"Found {len(bets)} bet(s) ({detected_format}):\n"
            for b in bets:
                 feedback_text += f"• {b.get('team','N/A')} ({b.get('odds',0):.2f})\n  Stake: €{b.get('stake',0):.2f} Time: {b.get('datetime', 'N/A')}\n"
            feedback_label.config(text=feedback_text)
            return bets
        else:
            logging.warning(f"No valid bets found using {detected_format} parser.")
            preview = "\n".join(lines[:15])
            feedback_label.config(text=f"No valid bets found ({detected_format}). Preview:\n{preview}")
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
def update_history_table(filtered_df=None):
    """Updates the Treeview with bet history, applying styles and formatting."""
    for item in tree.get_children():
        tree.delete(item)

    # Determine the DataFrame to display (either full history or filtered)
    display_df = filtered_df if filtered_df is not None else history.copy()

    # --- Filter to show only today's bets in the Treeview --- 
    if filtered_df is None: # Only apply daily filter if not already filtered (e.g., by league)
        try:
            # Ensure Date_dt column exists or create it for filtering
            if 'Date_dt' not in display_df.columns:
                if 'Date' in display_df.columns:
                    display_df['Date_dt'] = pd.to_datetime(display_df['Date'], errors='coerce', format='mixed')
                    if display_df['Date_dt'].isnull().all():
                        logging.warning("Created 'Date_dt' column, but all values are NaT. Cannot filter by day.")
                    else:
                         logging.debug("Created 'Date_dt' column for daily filtering.")
                else:
                    logging.warning("Cannot filter by day: 'Date'/'Date_dt' columns missing.")
                    # Proceed without filtering if no date column exists

            # Apply the filter if Date_dt is usable
            if 'Date_dt' in display_df.columns and not display_df['Date_dt'].isnull().all():
                today_date = pd.Timestamp.now().normalize() # Get today's date (midnight)
                # Keep rows where the date part of Date_dt matches today
                display_df = display_df[display_df['Date_dt'].dt.normalize() == today_date].copy() # Use .copy() to avoid SettingWithCopyWarning
                logging.info(f"Filtered Treeview to show only bets from {today_date.strftime('%Y-%m-%d')}. {len(display_df)} rows displayed.")
            # else: Logging handles cases where filtering isn't possible

        except Exception as e:
            logging.error(f"Error during daily filtering in update_history_table: {e}. Displaying unfiltered data for safety.")
    # --- End Daily Filter ---

    # --- Safely create Date_dt for sorting (if not already created/present) --- 
    # Sort ONLY by Date_dt (descending) to preserve screenshot order
    sort_columns = [] 
    sort_ascending = []
    
    if 'Date' in display_df.columns:
        try:
            # Attempt to convert 'Date' to datetime objects
            # Try parsing with milliseconds first, then fallback
            try:
                date_dt_col = pd.to_datetime(display_df['Date'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
                logging.debug("Parsed Date column with milliseconds format.")
            except (ValueError, TypeError):
                 logging.debug("Millisecond format failed, trying mixed format for Date column.")
                 date_dt_col = pd.to_datetime(display_df['Date'], errors='coerce', format='mixed')
                 
            # Check if *any* dates were successfully parsed
            if not date_dt_col.isnull().all():
                display_df['Date_dt'] = date_dt_col
                # Sort ONLY by date (most recent first)
                sort_columns = ['Date_dt'] 
                sort_ascending = [False] 
                logging.debug("Sorting history table primarily by Date_dt (desc) to preserve screenshot order.")
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
        
        # Determine tags based on result
        tags = []
        # Now we can safely use 'result'
        if result == "Pending":
            tags.append('pending')
            payout_str = "€0.00"
        elif result == "Win":
            tags.append('win')
            payout_str = f"+€{payout:.2f}" if payout > 0 else f"+€{stake * odds:.2f}"
        elif result == "Loss":
            tags.append('loss')
            payout_str = f"-€{stake:.2f}"
        else: 
             tags.append('pending')
             payout_str = f"€{payout:.2f}"

        # Format values for display
        odds_str = f"{odds:.2f}"
        stake_str = f"€{stake:.2f}"
        action_text = "✅ / ❌" if result == "Pending" else ""
        delete_symbol = "🗑️"
        
        # Insert item, including the DataFrame index as the LAST value
        item_id = tree.insert("", "end", values=(
            match_name, league, odds_str, stake_str, result,
            payout_str, display_date, action_text, delete_symbol,
            df_index # <<< Add DataFrame index here
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
        
        update_history_table() # Refresh the table to show changes
        feedback_label.config(text=f"Updated: {match_name} - {result}. Bankroll: €{bankroll:.2f}")
        logging.info(f"Updated bet result: {match_name} - {result}. New Bankroll: €{bankroll:.2f}")
        
    except Exception as e:
        logging.error(f"Error updating bet result for {selected_item_id}: {e}")
        messagebox.showerror("Error", f"Failed to update bet result: {e}")

def delete_selected_bet(selected_item_id):
    """Deletes the selected bet from history using its DataFrame index."""
    # --- Enhanced Logging Start --- 
    logging.info(f"--- delete_selected_bet initiated for Treeview item: {selected_item_id} ---")
    global history, bankroll
    
    # Log initial history size
    logging.debug(f"  History size BEFORE delete attempt: {len(history)}")
    logging.debug(f"  History indices BEFORE delete attempt: {history.index.tolist()}")
    
    try:
        item_values = tree.item(selected_item_id)['values']
        logging.info(f"  Retrieved item values: {item_values}")
        
        # --- Retrieve DataFrame index from Treeview --- 
        df_index_to_delete = None # Initialize
        try:
            # Assuming 'Index' is the 10th value (index 9)
            df_index_to_delete = int(item_values[9]) 
            logging.info(f"  Extracted DataFrame index from Treeview: {df_index_to_delete}")
        except (IndexError, ValueError, TypeError) as e:
            logging.error(f"  ERROR retrieving DataFrame index (index 9) from Treeview values ({item_values}): {e}")
            messagebox.showerror("Delete Error", "Could not identify the bet's internal ID.")
            return

        # --- Verify index exists in current history DataFrame --- 
        index_exists = df_index_to_delete in history.index
        logging.info(f"  Checking if index {df_index_to_delete} exists in history.index: {index_exists}")
        if not index_exists:
            logging.error(f"  DataFrame index {df_index_to_delete} NOT FOUND in current history DataFrame indices: {history.index.tolist()}")
            messagebox.showerror("Delete Error", "Bet not found in history (it might have been deleted already or data is out of sync).")
            update_history_table() # Refresh table to reflect current state
            return

        # --- Get necessary data directly from DataFrame using the index --- 
        logging.debug(f"  Accessing history.loc[{df_index_to_delete}]...")
        original_row = history.loc[df_index_to_delete]
        original_result = original_row["Result"]
        original_payout = float(original_row["Payout"]) # Ensure float
        original_stake = float(original_row["Stake"])   # Ensure float
        match_name = original_row["Match"] # Get name for feedback msg
        logging.info(f"  Data for index {df_index_to_delete}: Result={original_result}, Payout={original_payout}, Stake={original_stake}")

        # --- Reverse bankroll impact ONLY if bet was completed --- 
        # (Bankroll adjustment logic remains the same)
        if original_result == "Win":
            bankroll -= original_payout 
            logging.info(f"Deleting Won Bet: Reversing bankroll impact by -€{original_payout:.2f}")
        elif original_result == "Loss":
            bankroll += original_stake 
            logging.info(f"Deleting Lost Bet: Reversing bankroll impact by +€{original_stake:.2f}")
        else: 
             logging.info(f"Deleting Pending/Other status bet ({original_result}). No bankroll adjustment needed.")
    
        # --- Remove from history DataFrame using the direct index --- 
        logging.info(f"  Attempting to drop index {df_index_to_delete} from history...")
        history = history.drop(index=df_index_to_delete).reset_index(drop=True)
        logging.info(f"  Index {df_index_to_delete} dropped. New history size: {len(history)}")
        logging.debug(f"  History indices AFTER drop: {history.index.tolist()}") # Should be renumbered from 0
        
        # --- Save changes --- 
        logging.debug("  Saving bankroll and history...")
        save_bankroll(bankroll)
        save_history(history)
        logging.debug("  Save complete.")
        
        # --- Update UI --- 
        logging.debug("  Updating UI (bankroll label, history table, feedback label)...")
        br_label.config(text=f"Bankroll: €{bankroll:.2f}")
        update_history_table() # Refresh table view
        feedback_label.config(text=f"Deleted bet: {match_name}")
        logging.info(f"--- Successfully deleted bet: {match_name} (Index was: {df_index_to_delete}). Bankroll: €{bankroll:.2f} ---")
        
    except KeyError as ke:
         # Specific handling if .loc fails even after index check (shouldn't happen often)
         logging.error(f"KeyError accessing history.loc[{df_index_to_delete}] even after index check: {ke}", exc_info=True)
         messagebox.showerror("Delete Error", f"Error accessing bet data (Index: {df_index_to_delete}). Data might be inconsistent.")
    except Exception as e:
        logging.error(f"--- ERROR during delete_selected_bet for item {selected_item_id} (index {df_index_to_delete}): {e} ---", exc_info=True)
        messagebox.showerror("Error", f"Failed to delete bet: {e}")

def on_tree_click(event):
    """Handles left-clicks on the Treeview, specifically for action/delete cells."""
    # --- VERY FIRST LOG: Check if function is called at all --- 
    logging.info(f"*** on_tree_click function CALLED at ({event.x}, {event.y}) ***")
    
    # ... (rest of the function: logging, identification, logic) ...
    logging.debug(f"on_tree_click triggered at ({event.x}, {event.y})")
    region = tree.identify("region", event.x, event.y)
    column_id = tree.identify_column(event.x)
    selected_item_id = tree.identify_row(event.y)
    logging.debug(f"  Identified region: {region}, item: {selected_item_id}, column_id: {column_id}")
    
    if region == "cell" and selected_item_id and column_id:
        try:
             column_index = int(column_id.replace('#', '')) - 1
             logging.debug(f"  Calculated column index: {column_index}")
        except ValueError:
             logging.warning("  Could not determine column index from column_id.")
             return
             
        if not selected_item_id: return # No item clicked
        
        # --- Get item values early for debugging --- 
        item_values = None
        try:
            item_values = tree.item(selected_item_id)['values']
            logging.debug(f"  Retrieved item values for {selected_item_id}: {item_values}")
        except Exception as e:
             logging.error(f"  Failed to get item values for {selected_item_id}: {e}")
             return # Cannot proceed without item values
        
        # Check if item_values is valid (not None and not empty)
        if not item_values:
            logging.warning(f"  Item values list is empty or None for {selected_item_id}. Aborting click action.")
            return
        
        # --- Check for Delete Column Click FIRST --- 
        if column_index == 8:  # Delete column (index 8)
            logging.info(f"--> Delete column ({column_index}) clicked for item: {selected_item_id}")
            # --- Log the values just before calling delete --- 
            logging.info(f"    Values associated with item {selected_item_id}: {item_values}")
            # Check if index exists at position 9 before calling delete
            if len(item_values) > 9:
                 logging.info(f"    Index value at item_values[9]: {item_values[9]}")
            else:
                 logging.warning(f"    Item values list has length {len(item_values)}, expected > 9. Index might be missing.")
                 
            delete_selected_bet(selected_item_id)
            return # Action completed, exit function

        # --- Check for Actions Column Click (Only if Result is Pending) --- 
        if column_index == 7: # Actions column (index 7)
            try:
                current_result = item_values[4] # Get result from values
            except IndexError:
                logging.error(f"Could not get result (index 4) from item_values: {item_values}")
                return
                
            if current_result == "Pending":
                logging.debug("  Click detected in Actions column for Pending bet.")
                cell_box = tree.bbox(selected_item_id, column=column_id)
                if cell_box:
                     relative_x = event.x - cell_box[0]
                     # Determine if click was on Win (left) or Loss (right) part
                     if relative_x < cell_box[2] / 2:
                         logging.debug(f"  Calling update_bet_result (Win) for item: {selected_item_id}")
                         update_bet_result(selected_item_id, "Win")
                     else:
                         logging.debug(f"  Calling update_bet_result (Loss) for item: {selected_item_id}")
                         update_bet_result(selected_item_id, "Loss")
                     return # Action completed, exit function
                else: 
                     logging.warning("  Could not get cell bbox for Actions column.")
            else:
                logging.debug(f"  Click in Actions column ignored (Result is '{current_result}')")
        # --- If click was not in Delete or actionable Actions column --- 
        else:
            logging.debug(f"  Click was in column {column_index}, not actionable.")

def show_context_menu(event):
    """Displays a right-click context menu for the selected Treeview item."""
    logging.debug(f"show_context_menu triggered at ({event.x_root}, {event.y_root})") # Log entry
    selected_item_id = tree.identify_row(event.y)
    logging.debug(f"  Identified item for context menu: {selected_item_id}") # Log item
    if selected_item_id:
        tree.selection_set(selected_item_id) # Select the row under the cursor
        menu = tk.Menu(root, tearoff=0, bg="#424242", fg="#E0E0E0", 
                       activebackground="#616161", activeforeground="#FFFFFF")
        item_values = tree.item(selected_item_id)['values']
        result = item_values[4]
        
        if result == "Pending":
            # Log command calls
            menu.add_command(label="Mark as Win", 
                           command=lambda id=selected_item_id: (logging.debug(f"Context menu: Calling update_bet_result({id}, Win)"), update_bet_result(id, "Win")))
            menu.add_command(label="Mark as Loss", 
                           command=lambda id=selected_item_id: (logging.debug(f"Context menu: Calling update_bet_result({id}, Loss)"), update_bet_result(id, "Loss")))
            menu.add_separator()
        
        # Remove confirmation from context menu delete action
        menu.add_command(label="Delete Bet", 
                        command=lambda id=selected_item_id: (logging.debug(f"Context menu: Calling delete_selected_bet({id})"), delete_selected_bet(id)))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

# --- Functionality for Stats Page ---
# Define these functions *before* they are used in button commands

def show_main_page():
    """Hides stats frame and shows the main betting interface frames."""
    logging.info("Switching to Main page.")
    if stats_frame.winfo_exists() and stats_frame.winfo_manager() == 'grid':
        stats_frame.grid_forget()
    
    # Show the main card frames
    br_frame.grid(row=0, column=1, sticky="ew", padx=20, pady=(10, 5))
    feedback_frame.grid(row=1, column=1, sticky="ew", padx=20, pady=5)
    league_filter_frame.grid(row=2, column=1, sticky="ew", padx=20, pady=5)
    history_frame.grid(row=3, column=1, sticky="nsew", padx=20, pady=(5, 10))

def update_stats_display(period='all'):
    """Calculates and displays statistics for the selected period."""
    logging.info(f"Updating stats display for period: {period}")
    
    # Initialize df_filtered to None
    df_filtered = None
    
    # Find the content frame within stats_frame 
    stats_content_frame = None
    if not stats_frame.winfo_exists(): # Ensure stats frame exists first
        logging.error("Stats frame does not exist when trying to update stats display.")
        return
        
    for child in stats_frame.winfo_children():
         # Assuming the content frame is the one with bg #313131 and is a Frame
        if isinstance(child, tk.Frame) and child.cget('bg') == "#313131": 
             stats_content_frame = child
             break
    
    if not stats_content_frame:
        logging.error("Could not find stats_content_frame to display stats.")
        return 
        
    # Clear previous stats content
    for widget in stats_content_frame.winfo_children():
        widget.destroy()

    if history.empty:
        ttk.Label(stats_content_frame, text="No betting history to analyze.", 
                 font=("Segoe UI", 12, "italic"), background="#313131", foreground="#BDBDBD").pack(pady=20)
        logging.warning("Attempted to update stats, but history is empty.")
        return

    # --- Ensure Date column is datetime type for comparison --- 
    try:
        temp_history = history.copy() # Work on a copy
        if 'Date_dt' not in temp_history.columns or temp_history['Date_dt'].isnull().all():
             # Attempt conversion only if needed
             if 'Date' in temp_history.columns:
                 temp_history['Date_dt'] = pd.to_datetime(temp_history['Date'], errors='coerce', format='mixed')
             else:
                 # If no Date column, we can't proceed with date-based filtering/stats reliable
                 logging.warning("Stats calculation stopped: Missing 'Date' column in history.")
                 # Display message and return
                 for widget in stats_content_frame.winfo_children(): widget.destroy()
                 ttk.Label(stats_content_frame, text="Missing 'Date' column in history data.", 
                           font=("Segoe UI", 12), background="#313131", foreground="#E57373").pack(pady=20)
                 return
                 
        # Drop rows where Date_dt could not be parsed BEFORE assigning to df_filtered
        df_filtered_temp = temp_history.dropna(subset=['Date_dt']).copy()
        
        if df_filtered_temp.empty and not history.empty:
            # Handle case where all rows are dropped after date parsing attempt
            logging.warning("All rows dropped after date conversion/dropna for stats.")
            for widget in stats_content_frame.winfo_children(): widget.destroy()
            ttk.Label(stats_content_frame, text="Could not parse dates for stats calculation.", 
                     font=("Segoe UI", 12), background="#313131", foreground="#E57373").pack(pady=20)
            return # df_filtered remains None
        else:
             # Assign successfully processed data to df_filtered
             df_filtered = df_filtered_temp 
             logging.debug("Successfully prepared df_filtered with Date_dt column.")

    except Exception as e:
         # Catch errors during date processing/dropna
         logging.error(f"Error preparing history dates for stats: {e}", exc_info=True)
         for widget in stats_content_frame.winfo_children(): widget.destroy()
         ttk.Label(stats_content_frame, text="Error processing dates in history. Check logs.", 
                   font=("Segoe UI", 12), background="#313131", foreground="#E57373").pack(pady=20)
         return # df_filtered remains None

    # --- Proceed only if df_filtered was successfully created ---
    if df_filtered is None:
        # This case should ideally be caught by returns above, but as a safeguard:
        logging.error("df_filtered is None before numeric conversion, halting stats update.")
        return
        
    # --- Create Numeric Stake/Payout Columns EARLY --- 
    # Now df_filtered is guaranteed to exist if we reach here
    try:
        # 1. Ensure columns are string type before replacing characters
        if 'Stake' in df_filtered.columns:
             df_filtered['Stake_Clean'] = df_filtered['Stake'].astype(str).str.replace('€', '', regex=False).str.strip()
        if 'Payout' in df_filtered.columns:
             df_filtered['Payout_Clean'] = df_filtered['Payout'].astype(str).str.replace('€', '', regex=False).str.strip()

        # 2. Convert cleaned columns to numeric, coercing errors to NaN
        df_filtered['Stake_num'] = pd.to_numeric(df_filtered['Stake_Clean'], errors='coerce')
        df_filtered['Payout_num'] = pd.to_numeric(df_filtered['Payout_Clean'], errors='coerce')

        # 3. Log rows where conversion failed (resulted in NaN)
        stake_nan_rows = df_filtered[df_filtered['Stake_num'].isna()]
        payout_nan_rows = df_filtered[df_filtered['Payout_num'].isna()]
        if not stake_nan_rows.empty:
            logging.warning(f"Could not convert 'Stake' to numeric for rows (Index, Value): {list(zip(stake_nan_rows.index, stake_nan_rows['Stake']))}")
        if not payout_nan_rows.empty:
            logging.warning(f"Could not convert 'Payout' to numeric for rows (Index, Value): {list(zip(payout_nan_rows.index, payout_nan_rows['Payout']))}")

        # 4. Fill NaN values with 0 after logging, so calculations can proceed
        df_filtered['Stake_num'] = df_filtered['Stake_num'].fillna(0)
        df_filtered['Payout_num'] = df_filtered['Payout_num'].fillna(0)
        
        logging.debug("Cleaned and converted Stake/Payout to Stake_num/Payout_num, filling errors with 0.")
        
    except KeyError as e:
        # This error means the original 'Stake' or 'Payout' column is missing
        logging.error(f"Missing required column for numeric conversion: {e}")
        # Display an error message in the UI
        for widget in stats_content_frame.winfo_children(): widget.destroy() # Clear frame
        ttk.Label(stats_content_frame, text=f"Error: Missing data column '{e}' in history.", 
                   font=("Segoe UI", 12), background="#313131", foreground="#E57373").pack(pady=20)
        return
    except Exception as e:
        # Catch any other unexpected errors during cleaning/conversion
        logging.error(f"Unexpected error converting Stake/Payout to numeric: {e}", exc_info=True)
        for widget in stats_content_frame.winfo_children(): widget.destroy() # Clear frame
        ttk.Label(stats_content_frame, text="Error processing stake/payout data. Check logs.", 
                   font=("Segoe UI", 12), background="#313131", foreground="#E57373").pack(pady=20)
        return
    # --- END Numeric Column Creation ---

    # Filter data based on period
    current_date = pd.Timestamp.now().normalize() # Get date part only
    period_text = "All Time"
    if period == 'day':
        df_filtered = df_filtered[df_filtered['Date_dt'].dt.normalize() == current_date]
        period_text = "Today"
    elif period == 'week':
        start_date = current_date - pd.Timedelta(days=6) # Include today
        df_filtered = df_filtered[df_filtered['Date_dt'].dt.normalize() >= start_date]
        period_text = "Last 7 Days"
    elif period == 'month':
        start_date = current_date - pd.Timedelta(days=29) # Include today
        df_filtered = df_filtered[df_filtered['Date_dt'].dt.normalize() >= start_date]
        period_text = "Last 30 Days"
    
    if df_filtered.empty:
        ttk.Label(stats_content_frame, text=f"No bets found for: {period_text}", 
                 font=("Segoe UI", 12, "italic"), background="#313131", foreground="#BDBDBD").pack(pady=20)
        logging.info(f"No bets found for the period: {period_text}")
        return

    # Calculate stats
    total_bets = len(df_filtered)
    completed_bets_df = df_filtered[df_filtered["Result"] != "Pending"].copy()
    completed_bets_count = len(completed_bets_df)
    wins = len(completed_bets_df[completed_bets_df["Result"] == "Win"])
    losses = len(completed_bets_df[completed_bets_df["Result"] == "Loss"])
    pending = total_bets - completed_bets_count
    win_rate = (wins / completed_bets_count * 100) if completed_bets_count > 0 else 0
    # Remove creation here as it's done earlier
    # completed_bets_df.loc[:, 'Stake_num'] = pd.to_numeric(completed_bets_df['Stake'], errors='coerce').fillna(0)
    # completed_bets_df.loc[:, 'Payout_num'] = pd.to_numeric(completed_bets_df['Payout'], errors='coerce').fillna(0)
    # df_filtered.loc[:, 'Stake_num'] = pd.to_numeric(df_filtered['Stake'], errors='coerce').fillna(0)
    total_stake = df_filtered['Stake_num'].sum() # Use the pre-calculated column
    if not completed_bets_df.empty:
        # Use the pre-calculated columns here too
        profit_on_wins = completed_bets_df.loc[completed_bets_df["Result"] == "Win", 'Payout_num'].sum()
        stake_on_wins = completed_bets_df.loc[completed_bets_df["Result"] == "Win", 'Stake_num'].sum()
        stake_on_losses = completed_bets_df.loc[completed_bets_df["Result"] == "Loss", 'Stake_num'].sum()
        total_profit = (profit_on_wins - stake_on_wins) - stake_on_losses
    else:
        total_profit = 0.0
    roi = (total_profit / total_stake * 100) if total_stake > 0 else 0

    # --- Start Advanced Stats Calculations (Overall) ---
    avg_stake = total_stake / total_bets if total_bets > 0 else 0
    avg_odds_placed = df_filtered['Odds'].mean() if total_bets > 0 else 0
    
    biggest_win = 0
    biggest_loss = 0
    longest_win_streak = 0
    longest_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    profit_per_bet = [] # For variance/std dev

    if not completed_bets_df.empty:
        # Calculate profit/loss per completed bet
        completed_bets_df['ProfitLoss'] = completed_bets_df.apply(
            lambda row: (row['Payout_num'] - row['Stake_num']) if row['Result'] == 'Win' else -row['Stake_num'], axis=1
        )
        profit_per_bet = completed_bets_df['ProfitLoss'].tolist()
        biggest_win = completed_bets_df['ProfitLoss'].max()
        biggest_loss = completed_bets_df['ProfitLoss'].min() # Loss is negative

        # Sort by date to calculate streaks
        streaks_df = completed_bets_df.sort_values(by='Date_dt')
        for result in streaks_df['Result']:
            if result == 'Win':
                current_win_streak += 1
                current_loss_streak = 0
                longest_win_streak = max(longest_win_streak, current_win_streak)
            elif result == 'Loss':
                current_loss_streak += 1
                current_win_streak = 0
                longest_loss_streak = max(longest_loss_streak, current_loss_streak)
            # Ignore Pending for streaks

    # Variance/Std Dev
    profit_variance = np.var(profit_per_bet) if profit_per_bet else 0
    profit_std_dev = np.std(profit_per_bet) if profit_per_bet else 0
    # --- End Advanced Stats Calculations (Overall) ---


    # --- Display Stats ---
    # Clear previous content FIRST
    for widget in stats_content_frame.winfo_children():
        widget.destroy()
        
    # Configure grid layout for stats_content_frame
    stats_content_frame.grid_columnconfigure(0, weight=1) # Allow content to expand horizontally
    # Define row weights - give row 5 (league_tree_frame) the weight to expand vertically
    stats_content_frame.grid_rowconfigure(5, weight=1)

    # Use a consistent font and padding for stats labels - Adjusted padding again
    stat_font = ("Segoe UI", 11)
    label_padding = {'pady': 2, 'padx': (10, 5)} # Pad right of label
    value_padding = {'pady': 2, 'padx': (5, 10)} # Pad left of value
    profit_color = "#81C784" if total_profit >= 0 else "#E57373"

    # Grid Placement within stats_content_frame
    row_index = 0
    
    # Overall Title
    overall_title_label = ttk.Label(stats_content_frame, text=f"Statistics for: {period_text}",
                                      font=("Segoe UI", 14, "bold"), background="#313131", foreground="#03A9F4")
    # Use grid instead of pack
    overall_title_label.grid(row=row_index, column=0, pady=(10, 15), sticky="n")
    row_index += 1

    # --- Summary Stats Frame (Wins/Losses/Pending) --- Simplified to 2 columns ---
    summary_frame = tk.Frame(stats_content_frame, bg="#313131")
    summary_frame.grid(row=row_index, column=0, sticky="ew", padx=10, pady=5)
    summary_frame.grid_columnconfigure(0, weight=1) # Label column takes weight for alignment
    summary_frame.grid_columnconfigure(1, weight=1) # Value column takes weight
    
    # Place labels in col 0 (sticky e), values in col 1 (sticky w)
    ttk.Label(summary_frame, text="Total Bets:", font=stat_font, anchor="e").grid(row=0, column=0, sticky="e", **label_padding)
    ttk.Label(summary_frame, text=f"{total_bets}", font=stat_font, anchor="w").grid(row=0, column=1, sticky="w", **value_padding)
    ttk.Label(summary_frame, text="Completed:", font=stat_font, anchor="e").grid(row=1, column=0, sticky="e", **label_padding)
    ttk.Label(summary_frame, text=f"{completed_bets_count}", font=stat_font, anchor="w").grid(row=1, column=1, sticky="w", **value_padding)
    ttk.Label(summary_frame, text="Pending:", font=stat_font, anchor="e").grid(row=2, column=0, sticky="e", **label_padding)
    ttk.Label(summary_frame, text=f"{pending}", font=stat_font, anchor="w").grid(row=2, column=1, sticky="w", **value_padding)
    ttk.Label(summary_frame, text="Wins:", font=stat_font, anchor="e").grid(row=3, column=0, sticky="e", **label_padding)
    ttk.Label(summary_frame, text=f"{wins}", font=stat_font, anchor="w", foreground="#81C784").grid(row=3, column=1, sticky="w", **value_padding)
    ttk.Label(summary_frame, text="Losses:", font=stat_font, anchor="e").grid(row=4, column=0, sticky="e", **label_padding)
    ttk.Label(summary_frame, text=f"{losses}", font=stat_font, anchor="w", foreground="#E57373").grid(row=4, column=1, sticky="w", **value_padding)
    ttk.Label(summary_frame, text="Win Rate:", font=stat_font, anchor="e").grid(row=5, column=0, sticky="e", **label_padding)
    ttk.Label(summary_frame, text=f"{win_rate:.1f}%", font=stat_font, anchor="w").grid(row=5, column=1, sticky="w", **value_padding)
    row_index += 1

    # --- Financial Stats Frame (Stake/Profit/ROI) --- Simplified to 2 columns ---
    financial_frame = tk.Frame(stats_content_frame, bg="#313131")
    financial_frame.grid(row=row_index, column=0, sticky="ew", padx=10, pady=10)
    financial_frame.grid_columnconfigure(0, weight=1) # Label column
    financial_frame.grid_columnconfigure(1, weight=1) # Value column
    
    ttk.Label(financial_frame, text="Total Stake:", font=stat_font, anchor="e").grid(row=0, column=0, sticky="e", **label_padding)
    ttk.Label(financial_frame, text=f"€{total_stake:.2f}", font=stat_font, anchor="w").grid(row=0, column=1, sticky="w", **value_padding)
    ttk.Label(financial_frame, text="Total Profit:", font=stat_font, anchor="e").grid(row=1, column=0, sticky="e", **label_padding)
    ttk.Label(financial_frame, text=f"€{total_profit:+.2f}", font=stat_font, anchor="w", foreground=profit_color).grid(row=1, column=1, sticky="w", **value_padding)
    ttk.Label(financial_frame, text="ROI:", font=stat_font, anchor="e").grid(row=2, column=0, sticky="e", **label_padding)
    ttk.Label(financial_frame, text=f"{roi:.1f}%", font=stat_font, anchor="w", foreground=profit_color).grid(row=2, column=1, sticky="w", **value_padding)
    row_index += 1 

    # --- Advanced Overall Stats Frame --- Simplified to 2 columns ---
    adv_overall_frame = tk.Frame(stats_content_frame, bg="#313131")
    adv_overall_frame.grid(row=row_index, column=0, sticky="ew", padx=10, pady=5)
    adv_overall_frame.grid_columnconfigure(0, weight=1) # Label column
    adv_overall_frame.grid_columnconfigure(1, weight=1) # Value column
    adv_row = 0

    ttk.Label(adv_overall_frame, text="Avg Stake:", font=stat_font, anchor="e").grid(row=adv_row, column=0, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"€{avg_stake:.2f}", font=stat_font, anchor="w").grid(row=adv_row, column=1, sticky="w", **value_padding)
    adv_row += 1
    ttk.Label(adv_overall_frame, text="Avg Odds (Placed):", font=stat_font, anchor="e").grid(row=adv_row, column=0, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"{avg_odds_placed:.2f}", font=stat_font, anchor="w").grid(row=adv_row, column=1, sticky="w", **value_padding)
    adv_row += 1
    ttk.Label(adv_overall_frame, text="Biggest Win:", font=stat_font, anchor="e").grid(row=adv_row, column=0, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"€{biggest_win:+.2f}", font=stat_font, anchor="w", foreground="#81C784").grid(row=adv_row, column=1, sticky="w", **value_padding)
    adv_row += 1
    ttk.Label(adv_overall_frame, text="Biggest Loss:", font=stat_font, anchor="e").grid(row=adv_row, column=0, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"€{biggest_loss:.2f}", font=stat_font, anchor="w", foreground="#E57373").grid(row=adv_row, column=1, sticky="w", **value_padding)
    adv_row += 1
    ttk.Label(adv_overall_frame, text="Longest Win Streak:", font=stat_font, anchor="e").grid(row=adv_row, column=0, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"{longest_win_streak}", font=stat_font, anchor="w").grid(row=adv_row, column=1, sticky="w", **value_padding)
    adv_row += 1
    ttk.Label(adv_overall_frame, text="Longest Loss Streak:", font=stat_font, anchor="e").grid(row=adv_row, column=0, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"{longest_loss_streak}", font=stat_font, anchor="w").grid(row=adv_row, column=1, sticky="w", **value_padding)
    adv_row += 1
    ttk.Label(adv_overall_frame, text="Profit Std Dev:", font=stat_font, anchor="e").grid(row=adv_row, column=0, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"€{profit_std_dev:.2f}", font=stat_font, anchor="w").grid(row=adv_row, column=1, sticky="w", **value_padding)
    adv_row += 1

    row_index += 1

    # --- Per-League Statistics Calculation ---
    league_stats = [] # Initialize list to store results
    if 'League' not in df_filtered.columns:
        logging.warning(f"Cannot calculate per-league stats: 'League' column missing...")
    else:
        # Prepare grouping column
        df_filtered['League_Group'] = df_filtered['League'].fillna('Unknown').astype(str).str.strip()
        df_filtered.loc[df_filtered['League_Group'] == '', 'League_Group'] = 'Unknown'
        if not df_filtered.empty:
             grouped = df_filtered.groupby('League_Group')
             for league, group_df in grouped:
                # --- Calculate ALL stats for this league --- 
                lg_total_stake = 0 
                lg_total_profit = 0.0
                lg_avg_stake = 0.0
                lg_avg_odds = 0.0
                lg_biggest_win = 0.0
                lg_biggest_loss = 0.0
                lg_total_bets = len(group_df)
                lg_completed_bets_df = group_df[group_df["Result"] != "Pending"].copy()
                lg_completed_bets_count = len(lg_completed_bets_df)
                # ... [Wins/Losses/Pending/Win Rate calculation] ...
                lg_wins = 0; lg_losses = 0
                if not lg_completed_bets_df.empty:
                    lg_wins = len(lg_completed_bets_df[lg_completed_bets_df["Result"] == "Win"])
                    lg_losses = len(lg_completed_bets_df[lg_completed_bets_df["Result"] == "Loss"])
                lg_pending = lg_total_bets - lg_completed_bets_count
                lg_win_rate = (lg_wins / lg_completed_bets_count * 100) if lg_completed_bets_count > 0 else 0
                
                # Stake
                if 'Stake_num' in group_df.columns:
                      lg_total_stake = group_df['Stake_num'].sum()
                else:
                      logging.warning(f"League group '{league}' missing 'Stake_num' column.")
                lg_avg_stake = lg_total_stake / lg_total_bets if lg_total_bets > 0 else 0
                
                # Odds
                lg_avg_odds = group_df['Odds'].mean() if lg_total_bets > 0 else 0
                
                # Profit/Loss/ROI/MaxWin/MaxLoss
                if not lg_completed_bets_df.empty:
                    if 'Payout_num' in lg_completed_bets_df.columns and 'Stake_num' in lg_completed_bets_df.columns:
                         lg_completed_bets_df['ProfitLoss'] = lg_completed_bets_df.apply(
                             lambda row: (row['Payout_num'] - row['Stake_num']) if row['Result'] == 'Win' else -row['Stake_num'], axis=1
                         )
                         lg_total_profit = lg_completed_bets_df['ProfitLoss'].sum()
                         if not lg_completed_bets_df['ProfitLoss'].empty and lg_completed_bets_df['ProfitLoss'].notna().any():
                            lg_biggest_win = lg_completed_bets_df['ProfitLoss'].max()
                            lg_biggest_loss = lg_completed_bets_df['ProfitLoss'].min()
                         else: # Handle cases where ProfitLoss could be all NaN if source data was bad
                            lg_biggest_win = 0; lg_biggest_loss = 0
                    else:
                         logging.warning(f"League '{league}' missing Payout/Stake num for profit.")
                lg_roi = (lg_total_profit / lg_total_stake * 100) if lg_total_stake > 0 else 0
                
                # Append dictionary with all calculated stats for this league
                league_stats.append({
                    "League": league, "Total Bets": lg_total_bets, "Wins": lg_wins,
                    "Losses": lg_losses, "Win Rate": lg_win_rate, "Total Stake": lg_total_stake,
                    "Total Profit": lg_total_profit, "ROI": lg_roi, "Avg Stake": lg_avg_stake,
                    "Avg Odds": lg_avg_odds, "Biggest Win": lg_biggest_win, "Biggest Loss": lg_biggest_loss
                })
    # --- END Per-League Calculation Loop --- 

    # --- Display Per-League Stats in Treeview ---
    # Add Separator
    ttk.Separator(stats_content_frame, orient='horizontal').grid(row=row_index, column=0, sticky="ew", pady=15, padx=10)
    row_index += 1
    # Add Title
    ttk.Label(stats_content_frame, text="Stats by League",
              font=("Segoe UI", 13, "bold"), background="#313131", foreground="#03A9F4").grid(row=row_index, column=0, pady=(0, 5), sticky="n")
    row_index += 1
    
    league_tree_frame = tk.Frame(stats_content_frame, bg="#313131")
    league_tree_frame.grid(row=row_index, column=0, sticky="nsew", padx=10, pady=(0, 10)) # Increased bottom pady
    league_tree_frame.grid_rowconfigure(0, weight=1)
    league_tree_frame.grid_columnconfigure(0, weight=1)
    row_index += 1 

    league_columns = ("League", "Bets", "Wins", "Losses", "Win %", "Stake", "Profit", "ROI %", 
                      "Avg Stake", "Avg Odds", "Max Win", "Max Loss")
    league_tree = ttk.Treeview(league_tree_frame, columns=league_columns, show="headings", style="Treeview", height=8)
    # ... (Headings setup - check they match league_columns exactly) ...
    for col in league_columns: league_tree.heading(col, text=col) # Simpler heading setup
    # --- Column Widths (Adjust as needed) ---
    league_tree.column("League", width=180, anchor="w"); league_tree.column("Bets", width=50, anchor="center")
    league_tree.column("Wins", width=50, anchor="center"); league_tree.column("Losses", width=50, anchor="center")
    league_tree.column("Win %", width=70, anchor="e"); league_tree.column("Stake", width=100, anchor="e")
    league_tree.column("Profit", width=100, anchor="e"); league_tree.column("ROI %", width=70, anchor="e")
    league_tree.column("Avg Stake", width=80, anchor="e"); league_tree.column("Avg Odds", width=60, anchor="center")
    league_tree.column("Max Win", width=80, anchor="e"); league_tree.column("Max Loss", width=80, anchor="e")
    
    league_scrollbar = ttk.Scrollbar(league_tree_frame, orient="vertical", command=league_tree.yview)
    league_tree.configure(yscrollcommand=league_scrollbar.set)
    league_tree.grid(row=0, column=0, sticky="nsew"); league_scrollbar.grid(row=0, column=1, sticky="ns")

    # Populate the league Treeview from the calculated league_stats list
    if not league_stats:
        # ... (handle no data) ...
        try:
            league_tree.insert("", "end", values=("No league data for this period.",) + ("",)*(len(league_columns)-1), tags=('pending',))
            logging.info("Displayed 'No league data' message in league stats Treeview.")
        except Exception as e:
            logging.error(f"Failed to insert 'No league data' message into league_tree: {e}")
    else:
        # This block needs to be indented under the else
        league_stats_sorted = sorted(league_stats, key=lambda x: x["Total Profit"], reverse=True)
        logging.debug(f"Populating league_tree with sorted data ({len(league_stats_sorted)} leagues): {league_stats_sorted}")
        for i, stats in enumerate(league_stats_sorted):
            # Ensure stats dictionary has all keys expected by league_columns
            profit_str = f"{stats.get('Total Profit', 0.0):+.2f}"
            roi_str = f"{stats.get('ROI', 0.0):.1f}%"
            avg_stake_str = f"€{stats.get('Avg Stake', 0.0):.2f}"
            avg_odds_str = f"{stats.get('Avg Odds', 0.0):.2f}"
            max_win_str = f"€{stats.get('Biggest Win', 0.0):+.2f}"
            max_loss_str = f"€{stats.get('Biggest Loss', 0.0):.2f}"
            tags = ['win' if stats.get('Total Profit', 0) > 0 else ('loss' if stats.get('Total Profit', 0) < 0 else 'pending')]
            if i % 2 == 1: tags.append('striped')
            league_tree.insert("", "end", values=(
                stats.get("League", "N/A"), stats.get("Total Bets", 0),
                stats.get("Wins", 0), stats.get("Losses", 0),
                f"{stats.get('Win Rate', 0.0):.1f}%", f"€{stats.get('Total Stake', 0.0):.2f}",
                profit_str, roi_str, avg_stake_str, avg_odds_str,
                max_win_str, max_loss_str
            ), tags=tuple(tags))

    # --- Display Performance by Odds Range --- 
    # ... (Odds range calculations remain the same) ...
    ttk.Label(stats_content_frame, text="Performance by Odds Range",
              font=("Segoe UI", 13, "bold"), background="#313131", foreground="#03A9F4").grid(row=row_index, column=0, pady=(10, 5), sticky="n")
    row_index += 1
    odds_tree_frame = tk.Frame(stats_content_frame, bg="#313131")
    odds_tree_frame.grid(row=row_index, column=0, sticky="nsew", padx=10, pady=(0, 10)) # Increased bottom pady
    # ... (Odds tree setup and population remain the same) ...
    row_index += 1

    # --- Display Performance by Team --- 
    # ... (Team calculations remain the same) ...
    ttk.Label(stats_content_frame, text="Performance by Team",
              font=("Segoe UI", 13, "bold"), background="#313131", foreground="#03A9F4").grid(row=row_index, column=0, pady=(10, 5), sticky="n")
    row_index += 1
    team_tree_frame = tk.Frame(stats_content_frame, bg="#313131")
    team_tree_frame.grid(row=row_index, column=0, sticky="nsew", padx=10, pady=(0, 10))
    stats_content_frame.grid_rowconfigure(row_index, weight=1) # Make this last row expand
    # ... (Team tree setup and population remain the same) ...

    logging.info(f"Stats display updated successfully for period: {period}.")

def show_stats_page():
    """Hides main frames and shows the statistics frame."""
    global stats_frame 
    logging.info("Switching to Stats page.")
    
    # Hide ALL main card frames
    br_frame.grid_forget()
    feedback_frame.grid_forget()
    league_filter_frame.grid_forget()
    history_frame.grid_forget()
    
    # Show stats frame using grid in COLUMN 1, spanning ALL rows
    if not stats_frame.winfo_exists():
         stats_frame = tk.Frame(root, bg="#212121")
    stats_frame.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=10, pady=5) 
    stats_frame.grid_rowconfigure(2, weight=1) 
    stats_frame.grid_columnconfigure(0, weight=1)
    
    # Clear previous stats widgets (important!)
    for widget in stats_frame.winfo_children():
        widget.destroy()
    
    # --- Recreate the layout within stats_frame (using cards soon) ---
    back_button_frame = tk.Frame(stats_frame, bg="#212121")
    back_button_frame.grid(row=0, column=0, sticky="ew", pady=(5, 10))
    ttk.Button(back_button_frame, text="← Back to Bets", command=show_main_page, style="TButton").pack(side="left", padx=10)
    
    period_frame = tk.Frame(stats_frame, bg="#212121")
    period_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
    ttk.Label(period_frame, text="Select Period:", font=("Segoe UI", 11, "bold"), foreground="#03A9F4").pack(side="left", padx=(0, 10))
    ttk.Button(period_frame, text="Today", command=lambda: update_stats_display('day'), style="TButton").pack(side="left", padx=5)
    ttk.Button(period_frame, text="Last 7 Days", command=lambda: update_stats_display('week'), style="TButton").pack(side="left", padx=5)
    ttk.Button(period_frame, text="Last 30 Days", command=lambda: update_stats_display('month'), style="TButton").pack(side="left", padx=5)
    ttk.Button(period_frame, text="All Time", command=lambda: update_stats_display('all'), style="TButton").pack(side="left", padx=5)
    
    # --- Stats Content Frame (will hold the stat cards) ---
    # Give this the card background color
    stats_content_frame = tk.Frame(stats_frame, bg="#313131", bd=1, relief="ridge") 
    stats_content_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
    stats_content_frame.grid_columnconfigure(0, weight=1)
    # Row configuration will happen dynamically in update_stats_display

    # Delay the initial stats update 
    stats_content_frame.after(100, lambda: update_stats_display('all'))

# --- End of Stats Page Functionality ---


# --- GUI Setup ---
root = TkinterDnD.Tk()
logging.info("TkinterDnD root created.")
root.title("Betting Calculator")
root.geometry("1920x1080") # New Full HD size

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

root.configure(bg="#212121") # Darker background

# Variables
bankroll = load_bankroll()
history = load_history()

# Styles - Apply a more consistent dark theme
style = ttk.Style()
# Configure a theme (e.g., 'clam', 'alt', 'default', 'classic')
# 'clam' or 'alt' often work well for custom styling
try:
    style.theme_use('clam') 
    logging.info("Using 'clam' theme.")
except tk.TclError:
    logging.warning("Clam theme not available, using default.")
    style.theme_use('default')

# General widget styling
style.configure(".", 
                background="#212121", 
                foreground="#E0E0E0", # Light gray text
                font=("Segoe UI", 10)) 
style.configure("TButton", 
                font=("Segoe UI", 10), 
                padding=6,
                background="#424242", # Darker button
                foreground="#FFFFFF") # White text
style.map("TButton",
          background=[('active', '#616161')]) # Slightly lighter on hover/press

# Labels within the main background
style.configure("TLabel", 
                font=("Segoe UI", 11), 
                background="#212121", 
                foreground="#E0E0E0")
# Labels INSIDE cards/frames
style.configure("Card.TLabel", 
                font=("Segoe UI", 10), 
                background="#313131", # Card background
                foreground="#E0E0E0")
style.configure("CardTitle.TLabel", 
                font=("Segoe UI", 12, "bold"), 
                background="#313131", 
                foreground="#03A9F4") # Accent color title
style.configure("Feedback.TLabel", # Feedback specifically
                font=("Segoe UI", 10), 
                background="#313131", # Match card background
                foreground="#BDBDBD", 
                padding=(5, 5))

style.configure("TCombobox", 
                font=("Segoe UI", 10),
                background="#424242",
                foreground="#FFFFFF",
                fieldbackground="#424242",
                selectbackground="#616161",
                selectforeground="#FFFFFF")
style.map('TCombobox', fieldbackground=[('readonly', '#424242')])
style.map('TCombobox', selectbackground=[('readonly', '#616161')])
style.map('TCombobox', selectforeground=[('readonly', '#FFFFFF')])

# Configure Treeview colors and fonts for dark theme
style.configure("Treeview",
                font=("Segoe UI", 10),
                background="#313131", # Match card background
                foreground="#E0E0E0",
                fieldbackground="#313131",
                rowheight=28) 

style.configure("Treeview.Heading",
                font=("Segoe UI", 10, "bold"),
                background="#424242", # Header background
                foreground="#03A9F4", # Light blue header text
                padding=(8, 4),
                relief="flat") 
style.map("Treeview.Heading",
          background=[('active', '#525252')])

# Selected item style
style.map('Treeview',
          background=[('selected', '#515151')], 
          foreground=[('selected', '#FFFFFF')])

# Tag configurations (Foreground colors)
style.configure("win.Treeview", foreground="#81C784") 
style.configure("loss.Treeview", foreground="#E57373") 
style.configure("pending.Treeview", foreground="#E0E0E0") 
style.configure("striped.Treeview", background="#3A3A3A") # Stripe background

# Card Frame Style (using tk.Frame as base)
card_style = {
    "bg": "#313131",
    "bd": 1, # Subtle border
    "relief": "ridge" # Subtle border relief
}

# --- Main Frames Setup ---
# Replace direct frame creation with styled frames
br_frame = tk.Frame(root, **card_style)
# Remove upload frame for now, simplify drop target feedback
# upload_frame = tk.Frame(root, **card_style)
feedback_frame = tk.Frame(root, **card_style) # Frame for feedback label
league_filter_frame = tk.Frame(root, **card_style)
history_frame = tk.Frame(root, **card_style)
stats_frame = tk.Frame(root, bg="#212121") # Stats main container still uses root bg

# --- Bankroll Frame Content ---
br_frame.grid_columnconfigure((0, 4), weight=1) # Keep centering config
br_frame.grid_columnconfigure((1, 2, 3), weight=0)

br_label = ttk.Label(br_frame, 
                    text=f"Bankroll: €{bankroll:.2f}", 
                    foreground="#FFFFFF", # White text on card
                    background="#313131", # Match card background
                    font=("Segoe UI", 14, "bold"))
br_label.grid(row=0, column=1, columnspan=3, pady=(10, 10))

button_style = {"width": 15}
ttk.Button(br_frame, text="Reset Bankroll", 
           command=lambda: adjust_bankroll(200 - bankroll), 
           style="TButton",
           **button_style).grid(row=1, column=1, padx=5, pady=(5, 10))
ttk.Button(br_frame, text="Manual Change", 
           command=manual_bankroll_change,
           style="TButton",
           **button_style).grid(row=1, column=2, padx=5, pady=(5, 10))
ttk.Button(br_frame, text="Show Stats", 
           command=show_stats_page,
           style="TButton",
           **button_style).grid(row=1, column=3, padx=5, pady=(5, 10))

# --- Drop Target Info & Feedback Frame ---
# Make the entire root window the drop target
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop) 
logging.info("Root window registered as drop target.")

feedback_frame.grid_columnconfigure(0, weight=1)

upload_info_label = ttk.Label(feedback_frame, 
                              text="Drop Bet Screenshot Anywhere To Add", 
                              style="Card.TLabel", # Use card label style
                              font=("Segoe UI", 11, "italic"),
                              foreground="#BDBDBD")
upload_info_label.grid(row=0, column=0, pady=(5, 0), padx=10)

feedback_label = ttk.Label(feedback_frame, 
                           text="App ready. Drop a screenshot.", 
                           style="Feedback.TLabel", # Specific style for feedback
                           wraplength=500, 
                           anchor="center") 
feedback_label.grid(row=1, column=0, pady=(0, 5), padx=10, sticky="ew")

# --- League Filter Frame Content ---
league_filter_frame.grid_columnconfigure(1, weight=1) # Allow combobox to expand a bit

ttk.Label(league_filter_frame, 
          text="Filter by League:", 
          style="CardTitle.TLabel", # Use card title style (or similar)
          foreground="#FFFFFF", # White text on card
          font=("Segoe UI", 11, "bold")
         ).grid(row=0, column=0, padx=(10, 5), pady=10, sticky="w")

league_var = tk.StringVar(value="All Leagues")
all_leagues = ["All Leagues"] + sorted(list(history['League'].astype(str).unique()))
league_combo = ttk.Combobox(league_filter_frame, 
                            textvariable=league_var,
                            values=all_leagues,
                            state="readonly",
                            width=40, # Adjusted width
                            font=("Segoe UI", 10))
league_combo.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ew")

# --- History Frame Content ---
history_frame.grid_rowconfigure(1, weight=1) # Allow treeview row to expand
history_frame.grid_columnconfigure(0, weight=1) # Allow treeview col to expand

history_title = ttk.Label(history_frame, text="Bet History (Today)", style="CardTitle.TLabel")
history_title.grid(row=0, column=0, columnspan=2, pady=(5, 5), padx=10, sticky="nw")

tree = ttk.Treeview(history_frame, 
                    columns=("Match", "League", "Odds", "Stake", "Result", "Payout", "Date", "Actions", "Delete", "Index"), # Added Index column
                    show="headings",
                    displaycolumns=("Match", "League", "Odds", "Stake", "Result", "Payout", "Date", "Actions", "Delete"), # Hide Index column visually
                    height=15, 
                    style="Treeview")
# ... (Treeview headings and column config remains the same) ...

scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
tree.grid(row=1, column=0, sticky='nsew', padx=(10, 0), pady=(0, 10)) # Treeview takes row 1
scrollbar.grid(row=1, column=1, sticky='ns', padx=(0, 10), pady=(0, 10))

# Bind mouse clicks to functions
tree.bind('<Button-1>', on_tree_click)      # Left-click handler
logging.info("--- <Button-1> bound to on_tree_click for history Treeview ---") # <<< ADD THIS LOG
tree.bind('<Button-3>', show_context_menu) # Right-click handler (for context menu)
logging.info("--- <Button-3> bound to show_context_menu for history Treeview ---")

history_frame.grid_rowconfigure(1, weight=1) # Allow treeview row to expand
history_frame.grid_columnconfigure(0, weight=1) # Allow treeview col to expand


# Define tree-related functions after tree creation
# Add tags based on result in update_history_table function
# Example of how to add tags (will be done in update_history_table):
# tree.item(item_id, tags=('win',))


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
update_history_table() # Load initial history data
update_league_filter() # Update league filter options based on loaded history

# --- Start the main loop ---
root.mainloop()