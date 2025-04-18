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

def extract_bet_info(image_path):
    """Extracts bet information from an image using OCR (English Format).

    Parses text assuming the English format with 'Stake' label and amount below.
    Looks for lines like 'o Team Name Odds' to identify bets.
    Args:
        image_path (str): Path to the image file.
    Returns:
        list: A list of dictionaries, each containing info for one bet, or None.
    """
    if reader is None:
        logging.error("EasyOCR Reader was not initialized. Cannot process image.")
        feedback_label.config(text="OCR Engine Error (EasyOCR not initialized)")
        return None
        
    try:
        # Load image with PIL first
        img = Image.open(image_path)
        # Ensure image is in RGB format for consistency 
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logging.debug("Converted image to RGB format for EasyOCR.")
            
        # Convert PIL Image to NumPy array
        img_np = np.array(img) 
        logging.debug(f"Converted image '{os.path.basename(image_path)}' to NumPy array with shape: {img_np.shape}")

        # Read text using EasyOCR from the NumPy array
        # Pass the image data directly instead of the path
        result = reader.readtext(img_np, detail=0, paragraph=True)
        text = "\n".join(result) 
        
        logging.info(f"--- EasyOCR Output (from NumPy array) ---\\n{text}\\n--- End EasyOCR Output ---")

        # Keep the rest of the parsing logic, which operates on the 'text' variable
        # and splits it into 'lines'. EasyOCR's output structure might require
        # adjustments here if paragraph=True doesn't produce a similar enough structure.
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        logging.debug(f"Cleaned lines for parsing: {lines}")

        bets = []
        i = 0
        num_lines = len(lines)
        
        # --- Regex Patterns (English Format - Revised bet line) ---
        # Selection + Odds: Team Name, space(s), Odds (allowing text after odds)
        selected_bet_pattern = re.compile(r'^(.*?)\s+(\d+[,.]?\d*)\b.*?$', re.IGNORECASE)
        # Keywords to ignore if they appear in the potential team name part
        ignore_keywords = ["Stake", "To Return", "Cash Out", "Single", "Result", "Payout"]
        # Stake Label: Find 'Stake' case-insensitive
        stake_label_pattern = re.compile(r'Stake', re.IGNORECASE)
        # Amount Value: Find number like X.YY or X,YY
        amount_value_pattern = re.compile(r'(\d+[.,]\d+)') 
        # Date Pattern (e.g., Mon 14 Apr)
        date_pattern = re.compile(r'([A-Za-z]{3}\s+\d{1,2}\s+[A-Za-z]{3})', re.IGNORECASE) 
        # Time Pattern (HH:MM, HH.MM, HH,MM, HHMM)
        time_pattern = re.compile(r'(\d{2})[:.,]?(\d{2})$')

        # --- Parsing Loop (Revised bet identification) ---
        while i < num_lines:
            line = lines[i]
            logging.debug(f"Processing line {i}: '{line}'")
            processed_bet_on_this_line = False # Flag to check if we jumped 'i'

            selected_match = selected_bet_pattern.match(line)
            if selected_match:
                potential_team_name = selected_match.group(1).strip()
                is_ignored = any(keyword.lower() in potential_team_name.lower() for keyword in ignore_keywords)
                
                if not is_ignored:
                    team_name = potential_team_name
                    odds_str_raw = selected_match.group(2)
                    try: 
                        odds_str = odds_str_raw.replace(',', '.')
                        odds = 0.0 # Initialize odds

                        # --- Odds Correction Logic ---
                        # Check if it's likely an integer representation (e.g., "200" for 2.00)
                        if '.' not in odds_str and odds_str.isdigit():
                            odds_int = int(odds_str)
                            # Assume integers >= 101 and <= 2000 (representing 1.01 to 20.00) need conversion
                            if 101 <= odds_int <= 2000: 
                                odds = float(odds_int) / 100.0
                                logging.info(f"    Corrected integer odds: '{odds_str_raw}' -> {odds:.2f}")
                            else:
                                # Treat as regular float if outside correction range (but still likely error)
                                odds = float(odds_str) 
                        else:
                            # Parse as float directly if it contains a decimal or isn't purely digits
                            odds = float(odds_str)
                        # --- End Odds Correction ---

                        # --- Odds Validation ---
                        MIN_ODDS = 1.01
                        MAX_ODDS = 20.0 # Set a reasonable upper limit
                        if not (MIN_ODDS <= odds <= MAX_ODDS):
                            logging.warning(f"    Odds {odds:.2f} (from '{odds_str_raw}') is outside the valid range [{MIN_ODDS}-{MAX_ODDS}]. Skipping this bet.")
                            # Continue to the next line in the outer loop
                            i = j -1 # Adjust index to re-evaluate the line after the failed bet section
                            processed_bet_on_this_line = False # Mark as not processed to allow outer loop increment
                            continue # Skip the rest of the bet processing for this invalid odds line
                        # --- End Odds Validation ---

                        logging.info(f"Found Potential Bet: Team='{team_name}', Odds={odds:.2f}")
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
                            # Scan logic for stake, date, time (no changes)
                            if not found_stake: # Look for stake
                                stake_match = stake_label_pattern.search(scan_line)
                                if stake_match:
                                    # Look for amount *after* 'Stake' label on the same line or next
                                    search_text_for_amount = scan_line[stake_match.end():].strip()
                                    # If nothing after 'Stake', check next line
                                    if not search_text_for_amount and (j + 1 < search_end_idx):
                                         search_text_for_amount = lines[j+1].strip()
                                         # Check if the next line is just the amount
                                         if amount_value_pattern.fullmatch(search_text_for_amount):
                                              last_scanned_line_idx = j + 1 # Advance outer loop if we used next line
                                         else:
                                              search_text_for_amount = "" # Reset if next line isn't just amount
                                              
                                    amount_match = amount_value_pattern.search(search_text_for_amount)
                                    if amount_match:
                                        try:
                                            stake_str = amount_match.group(1).replace(',', '.')
                                            current_bet['stake'] = float(stake_str)
                                            found_stake = True
                                            logging.info(f"    Found Stake Amount: {current_bet['stake']}")
                                        except ValueError: 
                                             logging.warning(f"    Could not parse stake value from '{amount_match.group(1)}'")
                                    #else: # Don't log if stake amount not found yet, might be later
                                    #    logging.debug(f"    'Stake' label found, but no amount nearby in '{search_text_for_amount}'")
                            # ... (rest of the inner loop for date/time remains the same) ...
                            if not found_date: # Look for date
                                date_match = date_pattern.search(scan_line)
                                if date_match: current_bet['date'] = date_match.group(1); found_date = True; logging.info(f"    Found Date: {current_bet['date']}")
                            if not found_time: # Look for time
                                time_match = time_pattern.search(scan_line)
                                if time_match: current_bet['time'] = f"{time_match.group(1)}:{time_match.group(2)}"; found_time = True; logging.info(f"    Found Time: {current_bet['time']}")
                            j += 1
                        
                        # Finalize bet if stake found
                        if found_stake:
                            # ... (datetime parsing) ...
                            current_bet['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Placeholder
                            if found_date and found_time: # Try to parse if found
                                try:
                                    formatted_date_str = f"{current_bet['date']} {datetime.now().year} {current_bet['time']}"
                                    current_bet['datetime'] = datetime.strptime(formatted_date_str, '%a %d %b %Y %H:%M').strftime('%Y-%m-%d %H:%M:%S')
                                except Exception: pass # Use fallback on error
                                
                            logging.info(f"Adding complete bet: {current_bet}")
                            bets.append(current_bet)
                            i = last_scanned_line_idx # Jump outer loop index
                            processed_bet_on_this_line = True
                        else:
                            logging.warning(f"Discarding bet (Stake missing): Team='{team_name}'")
                    except ValueError: # Error parsing odds (should be less frequent now)
                        logging.warning(f"Could not parse odds '{odds_str_raw}' for '{team_name}' even after checks. Skipping.")
                # else: keyword ignored
                
            # Increment outer loop index ONLY if we didn't process a bet and jump 'i'
            if not processed_bet_on_this_line:
                i += 1
        
        if bets:
            logging.info(f"Successfully extracted {len(bets)} bets (EasyOCR): {bets}")
            feedback_text = f"Found {len(bets)} bet(s):\\n"
            for b in bets:
                 feedback_text += f"• {b.get('team','N/A')} ({b.get('odds',0):.2f})\\n  Stake: €{b.get('stake',0):.2f} Time: {b.get('datetime', 'N/A')}\\n"
            feedback_label.config(text=feedback_text)
            return bets
        else:
            preview = "\\n".join(lines[:15])
            feedback_label.config(text=f"No valid English format bets found. Preview:\\n{preview}")
            logging.warning(f"No valid bets found using EasyOCR. OCR text:\n{text}")
            return None

    except AttributeError as ae:
         # Catch the specific AttributeError we saw
         logging.error(f"AttributeError during EasyOCR processing (possibly image loading issue): {ae}")
         import traceback
         logging.error(traceback.format_exc())
         feedback_label.config(text=f"Error processing image format/data. Check logs.")
         return None
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"Error in extract_bet_info (EasyOCR): {e}\\n{error_details}")
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
            # Remove " or Draw" suffix if present
            # ... (cleaning logic remains the same) ...
            match_name_cleaned = match_name_raw
            if isinstance(match_name_raw, str) and match_name_raw.endswith(" or Draw"):
                 try:
                      match_name_cleaned = match_name_raw.removesuffix(" or Draw")
                      logging.debug(f"Removed ' or Draw' suffix from '{match_name_raw}' -> '{match_name_cleaned}'")
                 except AttributeError:
                      match_name_cleaned = match_name_raw.replace(" or Draw", "")
                      logging.debug(f"Replaced ' or Draw' in '{match_name_raw}' -> '{match_name_cleaned}' (using replace)")
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
    sort_columns = ['League', 'Match'] # Default sort columns
    sort_ascending = [True, True]
    
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
                # Prioritize sorting by date (most recent first)
                sort_columns.insert(0, 'Date_dt') 
                sort_ascending.insert(0, False) 
                logging.debug("Sorting history table by Date_dt (desc), League, Match.")
            else:
                 logging.warning("Could not parse any dates in 'Date' column for sorting. Sorting by League/Match.")
        except Exception as e:
            logging.warning(f"Error processing 'Date' column for sorting: {e}. Sorting by League/Match.")
    else:
        logging.debug("No 'Date' column found for sorting. Sorting by League/Match.")
        
    # Sort the DataFrame based on the determined columns
    try:
        display_df = display_df.sort_values(by=sort_columns, ascending=sort_ascending)
    except KeyError as ke:
        # Fallback if even League/Match columns are somehow missing
        logging.error(f"Sorting failed, missing key column: {ke}")
        # Don't sort if essential columns missing
    except Exception as e:
        logging.error(f"Unexpected error during sorting: {e}")

    # Add rows to Treeview
    for index, row in display_df.iterrows():
        tags = []
        result = row.get("Result", "Pending") # Default to Pending if missing
        stake = row.get("Stake", 0.0)
        payout = row.get("Payout", 0.0)
        odds = row.get("Odds", 0.0)
        match_name = row.get("Match", "N/A")
        league = row.get("League", "N/A")
        date_str = row.get("Date", "N/A") # Get original date string
        display_date = date_str # Default display
        try: # Format date for display if possible
             # Try standard format first
             dt_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
             display_date = dt_obj.strftime('%d %b %Y %H:%M')
        except (ValueError, TypeError): 
             try: # Try older format if first fails
                 dt_obj = datetime.strptime(date_str, '%d.%m.%Y')
                 display_date = dt_obj.strftime('%d %b %Y') # No time in old format
             except (ValueError, TypeError):
                 pass # Keep original string if all parsing fails

        # Determine tags based on result
        if result == "Pending":
            tags.append('pending')
            payout_str = "€0.00" # Pending bets have 0 payout initially
        elif result == "Win":
            tags.append('win')
            # Payout should be calculated correctly when marked as won
            payout_str = f"+€{payout:.2f}" if payout > 0 else f"+€{stake * odds:.2f}" # Show calculated if 0?
        elif result == "Loss":
            tags.append('loss')
            payout_str = f"-€{stake:.2f}" # Show stake loss
        else: # Handle other potential states or nulls
             tags.append('pending') # Default tag
             payout_str = f"€{payout:.2f}"

        # Format values for display
        odds_str = f"{odds:.2f}"
        stake_str = f"€{stake:.2f}"
        
        # Determine text/symbols for Action/Delete columns
        # Use emojis for pending actions
        action_text = "✅ / ❌" if result == "Pending" else ""
        delete_symbol = "🗑️" # Or use text like "Del"
        
        # Insert item into the tree with the new columns
        item_id = tree.insert("", "end", values=(
            match_name,
            league,
            odds_str,
            stake_str,
            result,
            payout_str,
            display_date, # Use formatted or original date string
            action_text,  
            delete_symbol 
        ), tags=tuple(tags)) 

        # Apply striping based on tree index 
        if tree.index(item_id) % 2 == 1:
             tree.item(item_id, tags=tags + ['striped'])

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
    """Deletes the selected bet from history and adjusts bankroll if needed."""
    logging.debug(f"delete_selected_bet called for item: {selected_item_id}") # Log entry
    global history, bankroll
    try:
        item_values = tree.item(selected_item_id)['values']
        logging.debug(f"  Item values retrieved: {item_values}") # Log values
        # Get data based on current column order: Match, League, Odds, Stake, Result, Payout, Date
        match_name = item_values[0]
        odds = float(item_values[2])
        stake = float(item_values[3].replace('€', ''))
        result = item_values[4]
        # Payout parsing needs care depending on format in table (+€X.XX or -€Y.YY)
        # For simplicity, recalculate impact based on result
        
        # Find the bet in history
        matching_bets = history[
            (history["Match"] == match_name) &
            (history["Odds"].round(2) == round(odds, 2)) &
            (history["Stake"].round(2) == round(stake, 2))
            # Add Result/Date check if needed for uniqueness
        ]
        logging.debug(f"  DataFrame search for bet to delete returned empty: {matching_bets.empty}") # Log search result
        
        if matching_bets.empty:
             messagebox.showerror("Delete Error", f"Could not find unique bet for {match_name} to delete.")
             logging.error(f"Could not find unique bet in history for deletion: {item_values}")
             return
             
        idx = matching_bets.index[0]
        original_result = history.loc[idx, "Result"]
        original_payout = history.loc[idx, "Payout"]
        original_stake = history.loc[idx, "Stake"]

        # Reverse bankroll impact ONLY if bet was completed
        if original_result == "Win":
            bankroll -= original_payout # Remove original payout
            logging.info(f"Deleting Won Bet: Reversing bankroll impact by -€{original_payout:.2f}")
        elif original_result == "Loss":
            bankroll += original_stake # Add back original stake
            logging.info(f"Deleting Lost Bet: Reversing bankroll impact by +€{original_stake:.2f}")
        # No bankroll change if deleting a Pending bet
        
        # Remove from history DataFrame
        history = history.drop(index=idx).reset_index(drop=True)
        
        # Save changes
        save_bankroll(bankroll)
        save_history(history)
        
        # Update UI
        br_label.config(text=f"Bankroll: €{bankroll:.2f}")
        update_history_table()
        feedback_label.config(text=f"Deleted bet: {match_name}")
        logging.info(f"Deleted bet: {match_name} (Index: {idx}). Bankroll adjusted to €{bankroll:.2f}")
        
    except Exception as e:
        logging.error(f"Error deleting bet {selected_item_id}: {e}")
        messagebox.showerror("Error", f"Failed to delete bet: {e}")

def on_tree_click(event):
    """Handles left-clicks on the Treeview, specifically for action/delete cells."""
    logging.debug(f"on_tree_click triggered at ({event.x}, {event.y})") # Log entry
    region = tree.identify("region", event.x, event.y)
    selected_item_id = tree.identify_row(event.y)
    column_id = tree.identify_column(event.x)
    logging.debug(f"  Identified region: {region}, item: {selected_item_id}, column_id: {column_id}") # Log identification
    
    if region == "cell" and selected_item_id and column_id:
        try:
             column_index = int(column_id.replace('#', '')) - 1
             logging.debug(f"  Calculated column index: {column_index}") # Log index
        except ValueError:
             logging.debug("  Could not determine column index.")
             return
             
        if not selected_item_id: return
        item_values = tree.item(selected_item_id)['values']
        if not item_values: return
        current_result = item_values[4]
        
        if column_index == 7 and current_result == "Pending":  # Actions column (index 7)
            logging.debug("  Click detected in Actions column for Pending bet.")
            cell_box = tree.bbox(selected_item_id, column=column_id)
            if cell_box:
                 relative_x = event.x - cell_box[0]
                 if relative_x < cell_box[2] / 2:
                     logging.debug("  Calling update_bet_result (Win)")
                     update_bet_result(selected_item_id, "Win")
                 else:
                     logging.debug("  Calling update_bet_result (Loss)")
                     update_bet_result(selected_item_id, "Loss")
            else: 
                 logging.debug("  Could not get cell bbox for Actions column.")
        elif column_index == 8:  # Delete column (index 8)
            logging.debug("  Click detected in Delete column.")
            # Directly call delete_selected_bet without asking for confirmation.
            logging.debug("  Calling delete_selected_bet (no confirmation)")
            delete_selected_bet(selected_item_id)
        else:
            logging.debug(f"  Click was in column {column_index}, not Actions(7) or Delete(8).")

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
    # Hide stats frame if it exists
    if stats_frame.winfo_exists() and stats_frame.winfo_manager() == 'grid':
        stats_frame.grid_forget()
    
    # Show main frames using grid in COLUMN 1
    # Ensure br_frame is shown
    if br_frame.winfo_exists():
         br_frame.grid(row=0, column=1, sticky="ew", padx=10, pady=(10, 5))
    # Show other main frames
    if upload_frame.winfo_exists():
        upload_frame.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
    if league_filter_frame.winfo_exists():
        league_filter_frame.grid(row=2, column=1, sticky="ew", padx=10, pady=5)
    if history_frame.winfo_exists():
        history_frame.grid(row=3, column=1, sticky="nsew", padx=10, pady=(5, 10))

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

    # Use a consistent font and padding for stats labels
    stat_font = ("Segoe UI", 11)
    label_padding = {'pady': 2, 'padx': (10, 2)} 
    value_padding = {'pady': 2, 'padx': (2, 10)}
    profit_color = "#81C784" if total_profit >= 0 else "#E57373" # Green for profit, red for loss

    # Grid Placement within stats_content_frame
    row_index = 0
    
    # Overall Title
    overall_title_label = ttk.Label(stats_content_frame, text=f"Statistics for: {period_text}",
                                      font=("Segoe UI", 14, "bold"), background="#313131", foreground="#03A9F4")
    # Use grid instead of pack
    overall_title_label.grid(row=row_index, column=0, pady=(10, 15), sticky="n")
    row_index += 1

    # --- Summary Stats Frame (Wins/Losses/Pending) ---
    summary_frame = tk.Frame(stats_content_frame, bg="#313131")
    summary_frame.grid(row=row_index, column=0, sticky="ew", padx=10, pady=5)
    # Adjust column configuration for tighter spacing
    summary_frame.grid_columnconfigure(0, weight=1) # Left space
    summary_frame.grid_columnconfigure(1, weight=0) # Label column
    summary_frame.grid_columnconfigure(2, weight=0) # Space between label/value (minimal)
    summary_frame.grid_columnconfigure(3, weight=0) # Value column
    summary_frame.grid_columnconfigure(4, weight=1) # Right space
    
    # Apply new padding and ensure sticky options
    ttk.Label(summary_frame, text="Total Bets:", font=stat_font, anchor="e").grid(row=0, column=1, sticky="e", **label_padding)
    ttk.Label(summary_frame, text=f"{total_bets}", font=stat_font, anchor="w").grid(row=0, column=3, sticky="w", **value_padding)
    ttk.Label(summary_frame, text="Completed:", font=stat_font, anchor="e").grid(row=1, column=1, sticky="e", **label_padding)
    ttk.Label(summary_frame, text=f"{completed_bets_count}", font=stat_font, anchor="w").grid(row=1, column=3, sticky="w", **value_padding)
    ttk.Label(summary_frame, text="Pending:", font=stat_font, anchor="e").grid(row=2, column=1, sticky="e", **label_padding)
    ttk.Label(summary_frame, text=f"{pending}", font=stat_font, anchor="w").grid(row=2, column=3, sticky="w", **value_padding)
    ttk.Label(summary_frame, text="Wins:", font=stat_font, anchor="e").grid(row=3, column=1, sticky="e", **label_padding)
    ttk.Label(summary_frame, text=f"{wins}", font=stat_font, anchor="w", foreground="#81C784").grid(row=3, column=3, sticky="w", **value_padding) # Green
    ttk.Label(summary_frame, text="Losses:", font=stat_font, anchor="e").grid(row=4, column=1, sticky="e", **label_padding)
    ttk.Label(summary_frame, text=f"{losses}", font=stat_font, anchor="w", foreground="#E57373").grid(row=4, column=3, sticky="w", **value_padding) # Red
    ttk.Label(summary_frame, text="Win Rate:", font=stat_font, anchor="e").grid(row=5, column=1, sticky="e", **label_padding)
    ttk.Label(summary_frame, text=f"{win_rate:.1f}%", font=stat_font, anchor="w").grid(row=5, column=3, sticky="w", **value_padding)
    row_index += 1

    # --- Financial Stats Frame (Stake/Profit/ROI) ---
    financial_frame = tk.Frame(stats_content_frame, bg="#313131")
    financial_frame.grid(row=row_index, column=0, sticky="ew", padx=10, pady=10)
    # Adjust column configuration for tighter spacing
    financial_frame.grid_columnconfigure((0, 4), weight=1) # Side spaces
    financial_frame.grid_columnconfigure((1, 2, 3), weight=0) # Content columns minimal weight
    
    # Apply new padding and ensure sticky options
    ttk.Label(financial_frame, text="Total Stake:", font=stat_font, anchor="e").grid(row=0, column=1, sticky="e", **label_padding)
    ttk.Label(financial_frame, text=f"€{total_stake:.2f}", font=stat_font, anchor="w").grid(row=0, column=3, sticky="w", **value_padding)
    ttk.Label(financial_frame, text="Total Profit:", font=stat_font, anchor="e").grid(row=1, column=1, sticky="e", **label_padding)
    ttk.Label(financial_frame, text=f"€{total_profit:+.2f}", font=stat_font, anchor="w", foreground=profit_color).grid(row=1, column=3, sticky="w", **value_padding) # Dynamic color
    ttk.Label(financial_frame, text="ROI:", font=stat_font, anchor="e").grid(row=2, column=1, sticky="e", **label_padding)
    ttk.Label(financial_frame, text=f"{roi:.1f}%", font=stat_font, anchor="w", foreground=profit_color).grid(row=2, column=3, sticky="w", **value_padding) # Dynamic color
    row_index += 1 

    # --- Advanced Overall Stats Frame ---
    adv_overall_frame = tk.Frame(stats_content_frame, bg="#313131")
    adv_overall_frame.grid(row=row_index, column=0, sticky="ew", padx=10, pady=5)
    # Adjust column configuration for tighter spacing
    adv_overall_frame.grid_columnconfigure((0, 4), weight=1) # Side spaces
    adv_overall_frame.grid_columnconfigure((1, 2, 3), weight=0) # Content columns minimal weight
    adv_row = 0

    # Apply new padding and ensure sticky options
    ttk.Label(adv_overall_frame, text="Avg Stake:", font=stat_font, anchor="e").grid(row=adv_row, column=1, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"€{avg_stake:.2f}", font=stat_font, anchor="w").grid(row=adv_row, column=3, sticky="w", **value_padding)
    adv_row += 1
    ttk.Label(adv_overall_frame, text="Avg Odds (Placed):", font=stat_font, anchor="e").grid(row=adv_row, column=1, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"{avg_odds_placed:.2f}", font=stat_font, anchor="w").grid(row=adv_row, column=3, sticky="w", **value_padding)
    adv_row += 1
    ttk.Label(adv_overall_frame, text="Biggest Win:", font=stat_font, anchor="e").grid(row=adv_row, column=1, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"€{biggest_win:+.2f}", font=stat_font, anchor="w", foreground="#81C784").grid(row=adv_row, column=3, sticky="w", **value_padding)
    adv_row += 1
    ttk.Label(adv_overall_frame, text="Biggest Loss:", font=stat_font, anchor="e").grid(row=adv_row, column=1, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"€{biggest_loss:.2f}", font=stat_font, anchor="w", foreground="#E57373").grid(row=adv_row, column=3, sticky="w", **value_padding)
    adv_row += 1
    ttk.Label(adv_overall_frame, text="Longest Win Streak:", font=stat_font, anchor="e").grid(row=adv_row, column=1, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"{longest_win_streak}", font=stat_font, anchor="w").grid(row=adv_row, column=3, sticky="w", **value_padding)
    adv_row += 1
    ttk.Label(adv_overall_frame, text="Longest Loss Streak:", font=stat_font, anchor="e").grid(row=adv_row, column=1, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"{longest_loss_streak}", font=stat_font, anchor="w").grid(row=adv_row, column=3, sticky="w", **value_padding)
    adv_row += 1
    ttk.Label(adv_overall_frame, text="Profit Std Dev:", font=stat_font, anchor="e").grid(row=adv_row, column=1, sticky="e", **label_padding)
    ttk.Label(adv_overall_frame, text=f"€{profit_std_dev:.2f}", font=stat_font, anchor="w").grid(row=adv_row, column=3, sticky="w", **value_padding)
    adv_row += 1

    row_index += 1 # Increment main row index after this frame

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
    
    # Hide ALL main content frames, including bankroll
    if br_frame.winfo_exists() and br_frame.winfo_manager() == 'grid':
        br_frame.grid_forget()
    if upload_frame.winfo_exists() and upload_frame.winfo_manager() == 'grid':
        upload_frame.grid_forget()
    if league_filter_frame.winfo_exists() and league_filter_frame.winfo_manager() == 'grid':
        league_filter_frame.grid_forget()
    if history_frame.winfo_exists() and history_frame.winfo_manager() == 'grid':
        history_frame.grid_forget()
    
    # Show stats frame using grid in COLUMN 1, spanning ALL rows (0-3)
    if not stats_frame.winfo_exists():
         logging.warning("stats_frame did not exist, recreating it.")
         stats_frame = tk.Frame(root, bg="#212121")
    # Span all 4 rows used by the main layout
    stats_frame.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=10, pady=5) 
    stats_frame.grid_rowconfigure(2, weight=1) # Row containing stats_content_frame should expand
    stats_frame.grid_columnconfigure(0, weight=1)
    
    # Clear previous stats widgets
    for widget in stats_frame.winfo_children():
        widget.destroy()
    
    # --- Create the layout within stats_frame ---
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
    stats_content_frame = tk.Frame(stats_frame, bg="#313131") 
    stats_content_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
    stats_content_frame.grid_columnconfigure(0, weight=1)
    stats_content_frame.grid_rowconfigure(0, weight=1) # Give row 0 weight for the test label

    # Remove temporary debug label
    # test_label = ttk.Label(stats_content_frame, text="Test Label - stats_content_frame is visible!", 
    #                        background="yellow", foreground="black", font=("Segoe UI", 16))
    # test_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    # END TEMPORARY DEBUG

    # Delay the initial stats update slightly to allow window layout processing
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
style.configure("TLabel", 
                font=("Segoe UI", 11), 
                background="#212121", 
                foreground="#E0E0E0")
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


# Configure custom style for feedback label (more subtle)
style.configure("Feedback.TLabel",
                font=("Segoe UI", 10),
                background="#212121",
                foreground="#BDBDBD", # Slightly dimmer feedback text
                padding=(5, 5))

# Configure Treeview colors and fonts for dark theme
style.configure("Treeview",
                font=("Segoe UI", 10),
                background="#313131", # Slightly lighter than main bg
                foreground="#E0E0E0",
                fieldbackground="#313131",
                rowheight=28) # Adjust row height if needed

style.configure("Treeview.Heading",
                font=("Segoe UI", 10, "bold"),
                background="#424242", # Header background
                foreground="#03A9F4", # Light blue header text
                padding=(8, 4),
                relief="flat") # Flat header look
style.map("Treeview.Heading",
          background=[('active', '#525252')])

# Selected item style
style.map('Treeview',
          background=[('selected', '#515151')], # Darker selection background
          foreground=[('selected', '#FFFFFF')])

# --- Tag configurations for Treeview rows (Win/Loss/Pending) ---
# Use background colors for better visual distinction
style.configure("win.Treeview", background="#2E7D32", foreground="#FFFFFF") # Darker Green background, White text
style.configure("loss.Treeview", background="#C62828", foreground="#FFFFFF") # Darker Red background, White text
# Pending and striped can keep the default background or a subtle variation
style.configure("pending.Treeview", background="#313131", foreground="#E0E0E0") 
style.configure("striped.Treeview", background="#3A3A3A") # Slightly lighter stripe

# Create all frames with the new dark background
br_frame = tk.Frame(root, bg="#212121")
upload_frame = tk.Frame(root, bg="#212121")
history_frame = tk.Frame(root, bg="#212121")
stats_frame = tk.Frame(root, bg="#212121")
league_filter_frame = tk.Frame(root, bg="#212121")

# Bankroll Frame setup 
br_label = ttk.Label(br_frame, 
                    text=f"Bankroll: €{bankroll:.2f}", 
                    foreground="#03A9F4", # Use theme accent color
                    font=("Segoe UI", 14, "bold"))
# Center the label by making it span the middle columns (1, 2, 3)
br_label.grid(row=0, column=1, columnspan=3, pady=(10, 5)) 

# Bankroll buttons (using ttk.Button now for consistent styling)
button_style = {
    "width": 15 # Slightly wider buttons
}

# Remove the +50 and -50 buttons
# ttk.Button(br_frame, text="+€50", 
#            command=lambda: adjust_bankroll(50),
#            style="TButton", # Apply ttk style
#            **button_style).grid(row=1, column=0, padx=5, pady=5)
# 
# ttk.Button(br_frame, text="-€50", 
#            command=lambda: adjust_bankroll(-50),
#            style="TButton",
#            **button_style).grid(row=1, column=1, padx=5, pady=5)

# Add remaining buttons in the middle columns (1, 2, 3)
ttk.Button(br_frame, text="Reset Bankroll", 
           command=lambda: adjust_bankroll(200 - bankroll), # Assuming 200 is default
           style="TButton",
           **button_style).grid(row=1, column=1, padx=5, pady=5) # Column 1

ttk.Button(br_frame, text="Manual Change", 
           command=manual_bankroll_change,
           style="TButton",
           **button_style).grid(row=1, column=2, padx=5, pady=5) # Column 2

# Add Show Stats button
ttk.Button(br_frame, text="Show Stats", 
           command=show_stats_page,
           style="TButton",
           **button_style).grid(row=1, column=3, padx=5, pady=5) # Column 3

# Configure br_frame columns to center the buttons
# Give empty columns 0 and 4 equal weight to push content to the middle
br_frame.grid_columnconfigure(0, weight=1)
br_frame.grid_columnconfigure(1, weight=0) # Button column
br_frame.grid_columnconfigure(2, weight=0) # Button column
br_frame.grid_columnconfigure(3, weight=0) # Button column
br_frame.grid_columnconfigure(4, weight=1)


# File Upload Area - No specific drop zone widget needed anymore
# Make the entire root window the drop target
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop) 
logging.info("Root window registered as drop target.")

# Simplified upload section label
upload_info_label = ttk.Label(upload_frame, 
                              text="Drop Bet Screenshot Anywhere", 
                              font=("Segoe UI", 12, "italic"),
                              foreground="#BDBDBD") # Dimmer text
upload_info_label.pack(pady=(5, 0))

# Feedback Label - stays in the upload frame for now
feedback_label = ttk.Label(upload_frame, 
                           text="App ready. Drop a screenshot.", 
                           style="Feedback.TLabel",
                           wraplength=500, # Adjust wrap length if needed
                           anchor="center") # Center the text
feedback_label.pack(pady=5, padx=10, fill="x")


# Add league filter 
ttk.Label(league_filter_frame, 
          text="Filter by League:", 
          font=("Segoe UI", 11, "bold"),
          foreground="#03A9F4").pack(side="left", padx=(20, 5), pady=5) # Added y-padding

league_var = tk.StringVar(value="All Leagues")
# Fetch leagues dynamically for the combobox
all_leagues = ["All Leagues"] + sorted(list(history['League'].astype(str).unique()))
league_combo = ttk.Combobox(league_filter_frame, 
                            textvariable=league_var,
                            values=all_leagues,
                            state="readonly",
                            width=35, # Adjusted width
                            font=("Segoe UI", 10)) # Explicit font
league_combo.pack(side="left", padx=5, pady=5)

# ... rest of filter_by_league function ...

# History Table
tree = ttk.Treeview(history_frame, 
                    columns=("Match", "League", "Odds", "Stake", "Result", "Payout", "Date", "Actions", "Delete"), # Added Actions, Delete
                    show="headings",
                    height=18, 
                    style="Treeview")

# Configure columns (Adjusting headings and widths)
tree.heading("Match", text="Match", anchor="w")
tree.heading("League", text="League", anchor="w")
tree.heading("Odds", text="Odds", anchor="center")
tree.heading("Stake", text="Stake (€)", anchor="e") 
tree.heading("Result", text="Result", anchor="center")
tree.heading("Payout", text="Payout (€)", anchor="e") 
tree.heading("Date", text="Date Added", anchor="center") 
tree.heading("Actions", text="Actions", anchor="center") # New heading
tree.heading("Delete", text="Del", anchor="center")     # New heading (shortened)

# Configure column widths 
tree.column("Match", width=300, anchor="w")
tree.column("League", width=250, anchor="w") # << Increased width from 180
tree.column("Odds", width=70, anchor="center")
tree.column("Stake", width=90, anchor="e")
tree.column("Result", width=90, anchor="center")
tree.column("Payout", width=100, anchor="e")
tree.column("Date", width=120, anchor="center")
tree.column("Actions", width=100, anchor="center") # Width for Win/Loss
tree.column("Delete", width=40, anchor="center")   # Width for Delete button

# Add scrollbar for the Treeview
scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)

# Remove padding around the tree, use frame padding instead
# tree.pack(fill="both", expand=True, padx=20, pady=10) 
tree.grid(row=0, column=0, sticky='nsew', padx=(10, 0), pady=10) # Use grid
scrollbar.grid(row=0, column=1, sticky='ns', padx=(0, 10), pady=10) # Place scrollbar next to tree

# Bind mouse clicks to functions
tree.bind('<Button-1>', on_tree_click)      # Left-click handler
tree.bind('<Button-3>', show_context_menu) # Right-click handler (for context menu)

history_frame.grid_rowconfigure(0, weight=1)
history_frame.grid_columnconfigure(0, weight=1)


# Define tree-related functions after tree creation
# Add tags based on result in update_history_table function
# Example of how to add tags (will be done in update_history_table):
# tree.item(item_id, tags=('win',))


# --- Layout Management ---

# Configure grid weights for main layout to center content but allow more width
root.grid_rowconfigure(0, weight=0) # Bankroll frame row
root.grid_rowconfigure(1, weight=0) # Upload/Stats row
root.grid_rowconfigure(2, weight=0) # Filter/Stats row
root.grid_rowconfigure(3, weight=1) # History/Stats row (expands vertically)

root.grid_columnconfigure(0, weight=1) # << Adjusted weight for less empty space
root.grid_columnconfigure(1, weight=10) # <<<< Give content column much more weight >>>>
root.grid_columnconfigure(2, weight=1) # << Adjusted weight for less empty space


# Place initial frames using grid for better control
# Place frames in the center column (column 1)
br_frame.grid(row=0, column=1, sticky="ew", padx=10, pady=(10, 5))
# Initially show the main page components (these will also use column 1 via show_main_page)
show_main_page() # This function needs modification to use column 1

# --- Initial Data Load ---
# Ensure initial calls happen after main layout setup
update_history_table() # Load initial history data
update_league_filter() # Update league filter options based on loaded history

# --- Start the main loop ---
root.mainloop()