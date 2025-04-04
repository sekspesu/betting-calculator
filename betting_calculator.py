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
from team_data import get_team_league, get_all_teams, get_league_teams
from difflib import SequenceMatcher

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
    """Extracts bet information from an image using OCR.

    Parses the text to find individual bet blocks and extracts:
    - Selected Team
    - Odds
    - Stake (Panusesumma)
    - Potential Win (Võidusumma)
    - Date and Time

    Args:
        image_path (str): Path to the image file.

    Returns:
        list: A list of dictionaries, each containing info for one bet, 
              or None if no valid bets are found or an error occurs.
    """
    try:
        img = Image.open(image_path)
        # Use PSM 4 for better block detection, keep Estonian support
        custom_config = r'--oem 3 --psm 4 -l est+eng' 
        text = pytesseract.image_to_string(img, config=custom_config)
        logging.info(f"--- OCR Raw Output ---\n{text}\n--- End OCR ---")

        text = text.replace('|', 'l')
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        logging.debug(f"Cleaned lines for parsing: {lines}")

        bets = []
        i = 0
        num_lines = len(lines)
        
        # --- Regex Patterns ---
        selected_bet_pattern = re.compile(r'^o\\s+([^\\d].*?)\\s+(\\d+[,\\.]\\d+)$') # Allow digits in team name now
        amount_labels_pattern = re.compile(r'Panusesumma.*Võidusumma', re.IGNORECASE)
        amount_values_pattern = re.compile(r'(\\d+[,\\.]\\d+)\\s*€')
        date_pattern = re.compile(r'([A-Z]\\s+\\d+\\s+[a-z]{3,})', re.IGNORECASE)
        time_pattern = re.compile(r'^(\\d{2}:\\d{2})$')

        while i < num_lines:
            line = lines[i]
            logging.debug(f"Processing line {i}: '{line}'")

            selected_match = selected_bet_pattern.match(line)
            if selected_match:
                team_name = selected_match.group(1).strip()
                odds_str = selected_match.group(2).replace(',', '.')
                try:
                    odds = float(odds_str)
                except ValueError:
                    logging.warning(f"Could not parse odds '{odds_str}' on line {i}. Skipping potential bet.")
                    i += 1
                    continue 
                    
                logging.info(f"Found Potential Bet Start: Team='{team_name}', Odds={odds}")
                current_bet = {'team': team_name, 'odds': odds}
                
                # Search subsequent lines for details
                found_amounts = False
                found_date = False
                found_time = False
                last_scanned_line_idx = i # Keep track of how far we scanned
                
                search_end_idx = min(i + 7, num_lines) # Limit search to next few lines
                j = i + 1
                while j < search_end_idx:
                    scan_line = lines[j]
                    logging.debug(f"  Scanning line {j}: '{scan_line}'")
                    last_scanned_line_idx = j

                    # Check for Amount Labels THEN Amounts on next line
                    if not found_amounts and amount_labels_pattern.search(scan_line):
                        logging.debug(f"    Found amount labels line at {j}.")
                        if j + 1 < num_lines: # Check if there is a next line
                            amounts_line = lines[j+1]
                            logging.debug(f"    Checking next line ({j+1}) for amounts: '{amounts_line}'")
                            amount_values = amount_values_pattern.findall(amounts_line)
                            if len(amount_values) >= 2:
                                try:
                                    stake_str = amount_values[0].replace(',', '.')
                                    potential_win_str = amount_values[1].replace(',', '.')
                                    current_bet['stake'] = float(stake_str)
                                    current_bet['potential_win'] = float(potential_win_str)
                                    logging.info(f"    Found Stake={current_bet['stake']}, Pot.Win={current_bet['potential_win']}")
                                    found_amounts = True
                                    j += 1 # IMPORTANT: Skip the amount values line in the next iteration
                                    last_scanned_line_idx = j # Update last scanned index
                                except ValueError:
                                     logging.warning(f"    Could not parse stake/win values: {amount_values}")
                            else:
                                logging.warning(f"    Found labels but < 2 amounts (€) on next line: {amount_values}")
                        else:
                            logging.warning(f"    Found amount labels line but no subsequent line exists.")

                    # Check for Date (only if not already found)
                    if not found_date:
                        date_match = date_pattern.search(scan_line)
                        if date_match:
                            current_bet['date'] = date_match.group(1)
                            logging.info(f"    Found Date part: {current_bet['date']}")
                            found_date = True
                            
                    # Check for Time (only if not already found)
                    if not found_time:
                        time_match = time_pattern.search(scan_line)
                        if time_match:
                            current_bet['time'] = time_match.group(1)
                            logging.info(f"    Found Time part: {current_bet['time']}")
                            found_time = True
                             
                    j += 1 # Move to next line in inner search loop

                # After scanning, finalize the bet if complete
                if found_amounts: # Require amounts to consider it a valid bet
                    # Parse datetime (using fallbacks)
                    if found_date and found_time:
                        try:
                            month_map = {'jaan': 'Jan', 'veebr': 'Feb', 'märts': 'Mar', 'apr': 'Apr', 'mai': 'May', 'juuni': 'Jun', 'juuli': 'Jul', 'aug': 'Aug', 'sept': 'Sep', 'okt': 'Oct', 'nov': 'Nov', 'dets': 'Dec'}
                            date_parts = current_bet['date'].split()
                            if len(date_parts) >= 3 and date_parts[2].lower() in month_map:
                                month_abbr = month_map[date_parts[2].lower()]
                                formatted_date_str = f"{date_parts[1]} {month_abbr} {datetime.now().year} {current_bet['time']}"
                                parsed_datetime = datetime.strptime(formatted_date_str, '%d %b %Y %H:%M')
                                current_bet['datetime'] = parsed_datetime.strftime('%Y-%m-%d %H:%M:%S')
                                logging.info(f"    Parsed datetime: {current_bet['datetime']}")
                            else:
                                raise ValueError(f"Could not map month or unexpected date format: {current_bet['date']}")
                        except Exception as e:
                            logging.error(f"    Error parsing date/time '{current_bet.get('date')} {current_bet.get('time')}': {e}")
                            current_bet['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Fallback
                    else:
                        current_bet['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Fallback
                        logging.warning(f"    Date/Time not fully found for bet {current_bet['team']}, using current time.")
                    
                    # Add the completed bet
                    logging.info(f"Adding complete bet: {current_bet}")
                    bets.append(current_bet)
                    # Advance outer loop index PAST the block we just scanned
                    i = last_scanned_line_idx 
                else:
                    logging.warning(f"Discarding incomplete bet (amounts not found) starting on line {i}: Team='{current_bet.get('team')}'")
                    # Let outer loop increment normally by 1

            # Increment outer loop index if we didn't jump it
            i += 1

        # --- End of while loop ---
        
        if bets:
            # (Feedback logic remains the same)
            logging.info(f"Successfully extracted {len(bets)} bets: {bets}")
            feedback_text = f"Found {len(bets)} bet(s):\\n"
            for b in bets:
                 feedback_text += f"• {b.get('team','N/A')} ({b.get('odds',0):.2f})\\n  Stake: €{b.get('stake',0):.2f}, Pot. Win: €{b.get('potential_win',0):.2f}\\n  Time: {b.get('datetime', 'N/A')}\\n"
            feedback_label.config(text=feedback_text)
            return bets
        else:
            # Provide more context if possible
             preview = "\\n".join(lines[:15])
             feedback_label.config(text=f"No valid bets found. Check log. OCR Preview:\\n{preview}")
             logging.warning(f"No valid complete bets found in screenshot. OCR text was:\\n{text}")
             return None

    except pytesseract.TesseractNotFoundError:
        logging.error(f"Tesseract not found at the specified path: {pytesseract.pytesseract.tesseract_cmd}")
        messagebox.showerror("Error", f"Tesseract OCR not found or path is incorrect.\nPlease check installation and path in script.\nPath: {pytesseract.pytesseract.tesseract_cmd}")
        feedback_label.config(text="Tesseract OCR Error. Check path.")
        return None
    except Exception as e:
        # Add more specific error logging
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"Error in extract_bet_info: {e}\n{error_details}")
        # Try to provide some context in the UI feedback
        feedback_text = f"Error processing image: {str(e)}. Check logs."
        if 'text' in locals() and text: # If OCR ran but parsing failed
             feedback_text += f"\nOCR Preview:\n{' '.join(text.split(' ')[:5])}"
        feedback_label.config(text=feedback_text)
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

    # Define tags (ensure styles are defined in the main GUI setup)
    # Example: tree.tag_configure('pending', foreground='#FFC107') etc.

    display_df = filtered_df if filtered_df is not None else history.copy() # Work on a copy

    # Attempt to convert 'Date' column to datetime objects for sorting
    # Handle potential errors gracefully
    if 'Date' in display_df.columns:
        try:
            # Try parsing common formats, including the ISO-like format we now save
            display_df['Date_dt'] = pd.to_datetime(display_df['Date'], errors='coerce', 
                                                   format='mixed', # Try multiple formats
                                                   dayfirst=False) # Assume YYYY-MM-DD or MM/DD/YYYY if ambiguous
        except Exception as e:
            logging.warning(f"Could not parse all dates in 'Date' column for sorting: {e}. Using string sort.")
            display_df['Date_dt'] = None # Ensure column exists

        # Fallback to original string if conversion failed for a row
        display_df['Date_dt'] = display_df['Date_dt'].fillna(pd.Timestamp.min) 
        
        # Sort: Prioritize by Date (most recent first), then League, then Match
        display_df = display_df.sort_values(by=['Date_dt', 'League', 'Match'], ascending=[False, True, True])
    else:
         # If no 'Date' column, sort by League and Match only
         display_df = display_df.sort_values(by=['League', 'Match'])

    # Add rows to Treeview
    for index, row in display_df.iterrows():
        tags = []
        result = row.get("Result", "Pending") # Default to Pending if missing
        stake = row.get("Stake", 0.0)
        payout = row.get("Payout", 0.0)
        odds = row.get("Odds", 0.0)
        match_name = row.get("Match", "N/A")
        league = row.get("League", "N/A")
        date_str = row.get("Date", "N/A") # Original date string for display

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

        # Add striping tag based on tree index (must be done after insert)
        # We need the actual index in the tree, not the DataFrame index
        # item_id = tree.insert(...) - then check tree.index(item_id)
        
        # Format values for display
        odds_str = f"{odds:.2f}"
        stake_str = f"€{stake:.2f}"
        
        # Format date string for better readability if it's a datetime string
        try:
             # Try parsing the standard format we save
             dt_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
             display_date = dt_obj.strftime('%d %b %Y %H:%M') # E.g., 05 Apr 2024 21:30
        except (ValueError, TypeError):
             # If it doesn't match, display as is (might be older format or partial)
             display_date = date_str


        # Insert item into the tree
        item_id = tree.insert("", "end", values=(
            match_name,
            league,
            odds_str,
            stake_str,
            result,
            payout_str,
            display_date # Use formatted date
        ), tags=tuple(tags)) # Ensure tags is a tuple

        # Apply striping based on tree index (must be done after insert)
        if tree.index(item_id) % 2 == 1: # Apply to odd rows (0-based index)
             tree.item(item_id, tags=tags + ['striped']) # Add striped tag


# --- Treeview Binding --- (Example - Assuming context menu exists)
# def on_tree_click(event): ... (handle clicks if needed)
# tree.bind('<Button-1>', on_tree_click) 
# tree.bind("<Button-3>", show_context_menu) # Assuming show_context_menu is defined


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
        # If not found, maybe it wasn't created yet or was destroyed.
        # Let's create it if it's missing inside show_stats_page instead.
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
    # else: period == 'all' uses the already filtered df_filtered
    
    if df_filtered.empty:
        ttk.Label(stats_content_frame, text=f"No bets found for: {period_text}", 
                 font=("Segoe UI", 12, "italic"), background="#313131", foreground="#BDBDBD").pack(pady=20)
        logging.info(f"No bets found for the period: {period_text}")
        return

    # Calculate stats
    total_bets = len(df_filtered)
    completed_bets_df = df_filtered[df_filtered["Result"] != "Pending"].copy() # Work on a copy
    completed_bets_count = len(completed_bets_df)
    wins = len(completed_bets_df[completed_bets_df["Result"] == "Win"])
    losses = len(completed_bets_df[completed_bets_df["Result"] == "Loss"])
    pending = total_bets - completed_bets_count
    
    win_rate = (wins / completed_bets_count * 100) if completed_bets_count > 0 else 0
    
    # Ensure Stake and Payout are numeric, coercing errors
    completed_bets_df.loc[:, 'Stake_num'] = pd.to_numeric(completed_bets_df['Stake'], errors='coerce').fillna(0)
    completed_bets_df.loc[:, 'Payout_num'] = pd.to_numeric(completed_bets_df['Payout'], errors='coerce').fillna(0)
    df_filtered.loc[:, 'Stake_num'] = pd.to_numeric(df_filtered['Stake'], errors='coerce').fillna(0)
    
    total_stake = df_filtered['Stake_num'].sum()
    
    if not completed_bets_df.empty:
        profit_on_wins = completed_bets_df.loc[completed_bets_df["Result"] == "Win", 'Payout_num'].sum()
        stake_on_wins = completed_bets_df.loc[completed_bets_df["Result"] == "Win", 'Stake_num'].sum()
        stake_on_losses = completed_bets_df.loc[completed_bets_df["Result"] == "Loss", 'Stake_num'].sum()
        total_profit = (profit_on_wins - stake_on_wins) - stake_on_losses # Profit from wins minus stake from losses
    else:
        total_profit = 0.0
    
    roi = (total_profit / total_stake * 100) if total_stake > 0 else 0

    # --- Display Stats --- 
    ttk.Label(stats_content_frame, text=f"Statistics for: {period_text}", 
              font=("Segoe UI", 14, "bold"), background="#313131", foreground="#03A9F4").pack(pady=(10, 15))

    summary_frame = tk.Frame(stats_content_frame, bg="#313131")
    summary_frame.pack(fill="x", padx=20, pady=5)
    performance_frame = tk.Frame(stats_content_frame, bg="#313131")
    performance_frame.pack(fill="x", padx=20, pady=5)
    financial_frame = tk.Frame(stats_content_frame, bg="#313131")
    financial_frame.pack(fill="x", padx=20, pady=5)

    # Summary Stats
    ttk.Label(summary_frame, text=f"Total Bets Placed:", font=("Segoe UI", 11), background="#313131").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(summary_frame, text=f"{total_bets}", font=("Segoe UI", 11, "bold"), background="#313131").grid(row=0, column=1, sticky="e", padx=5, pady=2)
    ttk.Label(summary_frame, text=f"Completed Bets:", font=("Segoe UI", 11), background="#313131").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(summary_frame, text=f"{completed_bets_count}", font=("Segoe UI", 11, "bold"), background="#313131").grid(row=1, column=1, sticky="e", padx=5, pady=2)
    ttk.Label(summary_frame, text=f"Pending Bets:", font=("Segoe UI", 11), background="#313131").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(summary_frame, text=f"{pending}", font=("Segoe UI", 11, "bold"), background="#313131").grid(row=2, column=1, sticky="e", padx=5, pady=2)
    summary_frame.grid_columnconfigure(1, weight=1)
    
    # Performance Stats
    ttk.Label(performance_frame, text=f"Wins:", font=("Segoe UI", 11), background="#313131", foreground="#81C784").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(performance_frame, text=f"{wins}", font=("Segoe UI", 11, "bold"), background="#313131", foreground="#81C784").grid(row=0, column=1, sticky="e", padx=5, pady=2)
    ttk.Label(performance_frame, text=f"Losses:", font=("Segoe UI", 11), background="#313131", foreground="#E57373").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(performance_frame, text=f"{losses}", font=("Segoe UI", 11, "bold"), background="#313131", foreground="#E57373").grid(row=1, column=1, sticky="e", padx=5, pady=2)
    ttk.Label(performance_frame, text=f"Win Rate (Completed Bets):", font=("Segoe UI", 11), background="#313131").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(performance_frame, text=f"{win_rate:.1f}%", font=("Segoe UI", 11, "bold"), background="#313131").grid(row=2, column=1, sticky="e", padx=5, pady=2)
    performance_frame.grid_columnconfigure(1, weight=1)

    # Financial Stats
    profit_color = "#81C784" if total_profit >= 0 else "#E57373"
    roi_color = "#81C784" if roi >= 0 else "#E57373"
    ttk.Label(financial_frame, text=f"Total Amount Staked:", font=("Segoe UI", 11), background="#313131").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(financial_frame, text=f"€{total_stake:.2f}", font=("Segoe UI", 11, "bold"), background="#313131").grid(row=0, column=1, sticky="e", padx=5, pady=2)
    ttk.Label(financial_frame, text=f"Net Profit/Loss (Completed Bets):", font=("Segoe UI", 11), background="#313131", foreground=profit_color).grid(row=1, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(financial_frame, text=f"€{total_profit:.2f}", font=("Segoe UI", 11, "bold"), background="#313131", foreground=profit_color).grid(row=1, column=1, sticky="e", padx=5, pady=2)
    ttk.Label(financial_frame, text=f"Return on Investment (ROI):", font=("Segoe UI", 11), background="#313131", foreground=roi_color).grid(row=2, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(financial_frame, text=f"{roi:.1f}%", font=("Segoe UI", 11, "bold"), background="#313131", foreground=roi_color).grid(row=2, column=1, sticky="e", padx=5, pady=2)
    financial_frame.grid_columnconfigure(1, weight=1)

    logging.info(f"Stats display updated successfully for period: {period}")

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
    
    # Clear previous stats widgets before drawing new ones
    for widget in stats_frame.winfo_children():
        widget.destroy()
    
    # --- Create the layout within stats_frame ---
    back_button_frame = tk.Frame(stats_frame, bg="#212121")
    back_button_frame.grid(row=0, column=0, sticky="ew", pady=(5, 10))
    ttk.Button(back_button_frame, text="← Back to Bets", 
               command=show_main_page, 
               style="TButton").pack(side="left", padx=10)

    period_frame = tk.Frame(stats_frame, bg="#212121")
    period_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
    
    ttk.Label(period_frame, 
              text="Select Period:", 
              font=("Segoe UI", 11, "bold"),
              foreground="#03A9F4").pack(side="left", padx=(0, 10))

    ttk.Button(period_frame, text="Today", 
               command=lambda: update_stats_display('day'), 
               style="TButton").pack(side="left", padx=5)
    ttk.Button(period_frame, text="Last 7 Days", 
               command=lambda: update_stats_display('week'), 
               style="TButton").pack(side="left", padx=5)
    ttk.Button(period_frame, text="Last 30 Days", 
               command=lambda: update_stats_display('month'), 
               style="TButton").pack(side="left", padx=5)
    ttk.Button(period_frame, text="All Time", 
               command=lambda: update_stats_display('all'), 
               style="TButton").pack(side="left", padx=5)

    # Frame to hold the actual stats labels/graphs - Ensure it's created here
    stats_content_frame = tk.Frame(stats_frame, bg="#313131") 
    stats_content_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
    stats_content_frame.grid_columnconfigure(0, weight=1)
    
    update_stats_display('all') 

# --- End of Stats Page Functionality ---


# GUI Setup
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
                    columns=("Match", "League", "Odds", "Stake", "Result", "Payout", "Date"), # Removed Actions/Delete for now, can be added back if needed
                    show="headings",
                    height=18, # Adjusted height based on window size
                    style="Treeview") # Apply the base Treeview style

# Configure columns (Adjusting headings and widths)
tree.heading("Match", text="Match", anchor="w")
tree.heading("League", text="League", anchor="w")
tree.heading("Odds", text="Odds", anchor="center")
tree.heading("Stake", text="Stake (€)", anchor="e") # Indicate currency, align right
tree.heading("Result", text="Result", anchor="center")
tree.heading("Payout", text="Payout (€)", anchor="e") # Indicate currency, align right
tree.heading("Date", text="Date Added", anchor="center") # New heading for date

# Configure column widths 
tree.column("Match", width=300, anchor="w")
tree.column("League", width=180, anchor="w")
tree.column("Odds", width=70, anchor="center")
tree.column("Stake", width=90, anchor="e") # Align stake to the right
tree.column("Result", width=90, anchor="center")
tree.column("Payout", width=100, anchor="e") # Align payout to the right
tree.column("Date", width=120, anchor="center")

# Add scrollbar for the Treeview
scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)

# Remove padding around the tree, use frame padding instead
# tree.pack(fill="both", expand=True, padx=20, pady=10) 
tree.grid(row=0, column=0, sticky='nsew', padx=(10, 0), pady=10) # Use grid
scrollbar.grid(row=0, column=1, sticky='ns', padx=(0, 10), pady=10) # Place scrollbar next to tree

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