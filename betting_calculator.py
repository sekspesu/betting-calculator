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
from datetime import datetime
from team_data import get_team_league, get_all_teams, get_league_teams
from difflib import SequenceMatcher
import numpy as np # <-- Import numpy

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
                        odds = float(odds_str)
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
                                    search_after_stake = scan_line[stake_match.end():]
                                    amount_match = amount_value_pattern.search(search_after_stake)
                                    if amount_match:
                                        try:
                                            stake_str = amount_match.group(1).replace(',', '.')
                                            current_bet['stake'] = float(stake_str)
                                            found_stake = True
                                            logging.info(f"    Found Stake Amount: {current_bet['stake']}")
                                        except ValueError: pass # Logged inside
                                    else: pass # Logged inside
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
                    except ValueError: # Error parsing odds
                        logging.warning(f"Could not parse odds '{selected_match.group(2)}' for '{team_name}'. Skipping.")
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
        for bet in extracted_bets:
            # Prepare data for add_bet or update_result
            # Use 'team' instead of 'match' from extraction
            match_name = bet.get('team', 'Unknown Match') 
            odds = bet.get('odds', 0.0)
            stake = bet.get('stake', 0.0)
            # Use the extracted datetime if available, otherwise use current time
            bet_datetime_str = bet.get('datetime', datetime.now().strftime('%Y-%m-%d %H:%M:%S')) 

            # Decision logic (Add vs Update) might need refinement.
            # For now, just try adding every extracted bet.
            # Update logic could be triggered by dropping a "result" screenshot later.
            add_bet(match_name, odds, stake, bet_datetime_str) 
            num_added += 1
            
            # Example: Update logic placeholder (needs trigger)
            # if is_result_screenshot(file_path): # Need a way to determine this
            #    update_result(file_path, match_name, odds, stake)
            #    num_updated += 1
            # else:
            #    add_bet(match_name, odds, stake, bet_datetime_str)
            #    num_added += 1

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
        action_text = "W / L" if result == "Pending" else ""
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
    # Check if stats_frame exists and is managed by grid before removing
    if stats_frame.winfo_exists() and stats_frame.winfo_manager() == 'grid':
        stats_frame.grid_forget()
    
    # Show main frames using grid in COLUMN 1
    if upload_frame.winfo_exists():
        upload_frame.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
    if league_filter_frame.winfo_exists():
        league_filter_frame.grid(row=2, column=1, sticky="ew", padx=10, pady=5)
    if history_frame.winfo_exists():
        history_frame.grid(row=3, column=1, sticky="nsew", padx=10, pady=(5, 10))

def update_stats_display(period='all'):
    """Calculates and displays statistics for the selected period."""
    logging.info(f"Updating stats display for period: {period}")
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

    # Ensure Date column is datetime type for comparison
    try:
        if 'Date_dt' not in history.columns or history['Date_dt'].isnull().all():
             history['Date_dt'] = pd.to_datetime(history['Date'], errors='coerce', format='mixed')
        df_filtered = history.dropna(subset=['Date_dt']).copy()
        if df_filtered.empty and not history.empty:
            logging.warning("All rows dropped after date conversion/dropna for stats.")
            ttk.Label(stats_content_frame, text="Could not parse dates for stats.", 
                     font=("Segoe UI", 12), background="#313131", foreground="#E57373").pack(pady=20)
            return
    except Exception as e:
         logging.error(f"Error preparing history dates for stats: {e}")
         ttk.Label(stats_content_frame, text="Error processing dates in history.", 
                   font=("Segoe UI", 12), background="#313131", foreground="#E57373").pack(pady=20)
         return

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
    completed_bets_df.loc[:, 'Stake_num'] = pd.to_numeric(completed_bets_df['Stake'], errors='coerce').fillna(0)
    completed_bets_df.loc[:, 'Payout_num'] = pd.to_numeric(completed_bets_df['Payout'], errors='coerce').fillna(0)
    df_filtered.loc[:, 'Stake_num'] = pd.to_numeric(df_filtered['Stake'], errors='coerce').fillna(0)
    total_stake = df_filtered['Stake_num'].sum()
    if not completed_bets_df.empty:
        profit_on_wins = completed_bets_df.loc[completed_bets_df["Result"] == "Win", 'Payout_num'].sum()
        stake_on_wins = completed_bets_df.loc[completed_bets_df["Result"] == "Win", 'Stake_num'].sum()
        stake_on_losses = completed_bets_df.loc[completed_bets_df["Result"] == "Loss", 'Stake_num'].sum()
        total_profit = (profit_on_wins - stake_on_wins) - stake_on_losses
    else:
        total_profit = 0.0
    roi = (total_profit / total_stake * 100) if total_stake > 0 else 0

    # --- Display Stats ---
    # Use a consistent font and padding for stats labels
    stat_font = ("Segoe UI", 11)
    label_padding = {'pady': 3, 'padx': 10}
    value_padding = {'pady': 3, 'padx': 5}
    profit_color = "#81C784" if total_profit >= 0 else "#E57373" # Green for profit, red for loss

    ttk.Label(stats_content_frame, text=f"Statistics for: {period_text}",
              font=("Segoe UI", 14, "bold"), background="#313131", foreground="#03A9F4").pack(pady=(10, 15))

    # --- Summary Stats Frame (Wins/Losses/Pending) ---
    summary_frame = tk.Frame(stats_content_frame, bg="#313131")
    summary_frame.pack(fill="x", padx=20, pady=5)
    summary_frame.grid_columnconfigure((0, 2, 4), weight=1) # Add spacing columns

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

    # --- Financial Stats Frame (Stake/Profit/ROI) ---
    financial_frame = tk.Frame(stats_content_frame, bg="#313131")
    financial_frame.pack(fill="x", padx=20, pady=10)
    financial_frame.grid_columnconfigure((0, 2, 4), weight=1) # Add spacing columns

    ttk.Label(financial_frame, text="Total Stake:", font=stat_font, anchor="e").grid(row=0, column=1, sticky="e", **label_padding)
    ttk.Label(financial_frame, text=f"€{total_stake:.2f}", font=stat_font, anchor="w").grid(row=0, column=3, sticky="w", **value_padding)

    ttk.Label(financial_frame, text="Total Profit:", font=stat_font, anchor="e").grid(row=1, column=1, sticky="e", **label_padding)
    ttk.Label(financial_frame, text=f"€{total_profit:+.2f}", font=stat_font, anchor="w", foreground=profit_color).grid(row=1, column=3, sticky="w", **value_padding) # Dynamic color

    ttk.Label(financial_frame, text="ROI:", font=stat_font, anchor="e").grid(row=2, column=1, sticky="e", **label_padding)
    ttk.Label(financial_frame, text=f"{roi:.1f}%", font=stat_font, anchor="w", foreground=profit_color).grid(row=2, column=3, sticky="w", **value_padding) # Dynamic color

    logging.info(f"Stats display updated successfully for period: {period}. TotalProfit: {total_profit:.2f}, ROI: {roi:.1f}%")
    # Log intermediate values for debugging if needed
    logging.debug(f"Stats Data ({period}): TotalBets={total_bets}, Completed={completed_bets_count}, Wins={wins}, Losses={losses}, Pending={pending}, TotalStake={total_stake:.2f}, TotalProfit={total_profit:.2f}")

def show_stats_page():
    """Hides main frames and shows the statistics frame."""
    global stats_frame 
    logging.info("Switching to Stats page.")
    # Hide main content frames
    if upload_frame.winfo_exists() and upload_frame.winfo_manager() == 'grid':
        upload_frame.grid_forget()
    if league_filter_frame.winfo_exists() and league_filter_frame.winfo_manager() == 'grid':
        league_filter_frame.grid_forget()
    if history_frame.winfo_exists() and history_frame.winfo_manager() == 'grid':
        history_frame.grid_forget()
    
    # Show stats frame using grid in COLUMN 1
    if not stats_frame.winfo_exists():
         logging.warning("stats_frame did not exist, recreating it.")
         stats_frame = tk.Frame(root, bg="#212121")
    stats_frame.grid(row=1, column=1, rowspan=3, sticky="nsew", padx=10, pady=5) 
    stats_frame.grid_rowconfigure(2, weight=1) 
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
    update_stats_display('all') 

# --- End of Stats Page Functionality ---


# --- GUI Setup ---
root = TkinterDnD.Tk()
logging.info("TkinterDnD root created.")
root.title("Betting Calculator")
root.geometry("1300x800")  # Increased window size slightly more
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
# These tags will be applied in the update_history_table function
style.configure("win.Treeview", background="#313131", foreground="#81C784") # Green for win
style.configure("loss.Treeview", background="#313131", foreground="#E57373") # Red for loss
style.configure("pending.Treeview", background="#313131", foreground="#E0E0E0") # Default for pending

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
tree.column("League", width=180, anchor="w")
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

# Configure grid weights for main layout to center content
root.grid_rowconfigure(0, weight=0) # Bankroll frame row
root.grid_rowconfigure(1, weight=0) # Upload/Stats row
root.grid_rowconfigure(2, weight=0) # Filter/Stats row
root.grid_rowconfigure(3, weight=1) # History/Stats row (expands vertically)

root.grid_columnconfigure(0, weight=1) # Empty left column (expands)
root.grid_columnconfigure(1, weight=0) # <<<< Main Content Column >>>>
root.grid_columnconfigure(2, weight=1) # Empty right column (expands)


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