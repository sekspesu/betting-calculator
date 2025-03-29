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

def process_file(file_path):
    if not file_path:
        return
    bet_info = extract_bet_info(file_path)
    if bet_info:
        for bet in bet_info:
            match, odds, stake = bet["match"], bet["odds"], bet["stake"]
            # Fix binary file reading
            with open(file_path, "rb") as f:
                file_content = f.read().decode('utf-8', errors='ignore').lower()
            if "Pending" in history["Result"].values and ("win" in file_content or "loss" in file_content):
                update_result(file_path, match, odds, stake)
            else:
                add_bet(match, odds, stake)
    else:
        feedback_label.config(text="Failed to extract bet info. Check screenshot format.")
        logging.warning("Failed to extract bet info from screenshot.")

def add_bet(match, odds, stake):
    global history
    # Fix DataFrame concatenation warning by creating a properly typed DataFrame
    new_bet = pd.DataFrame({
        "Match": [match],
        "Odds": [float(odds)],
        "Stake": [float(stake)],
        "Result": ["Pending"],
        "Payout": [0.0],
        "Date": [datetime.now().strftime('%d.%m.%Y')]
    })
    history = pd.concat([history, new_bet], ignore_index=True)
    save_history(history)
    update_history_table()
    feedback_label.config(text=f"Added: {match}\nOdds: {odds}, Stake: €{stake:.2f}")
    logging.info(f"Added bet: {match}, Odds: {odds}, Stake: €{stake:.2f}")

def update_result(file_path, match, odds, stake):
    with open(file_path, "rb") as f:
        text = f.read().decode('utf-8', errors='ignore').lower()
    result = "Win" if "win" in text else "Loss"
    global bankroll, history
    if "Pending" in history["Result"].values:
        idx = history[history["Result"] == "Pending"].index[0]
        if history.loc[idx, "Match"] == match:
            if result == "Win":
                payout = stake * odds
                bankroll += payout - stake
                history.loc[idx, "Payout"] = payout
            else:
                bankroll -= stake
                history.loc[idx, "Payout"] = 0
            history.loc[idx, "Result"] = result
            save_bankroll(bankroll)
            save_history(history)
            br_label.config(text=f"Bankroll: €{bankroll:.2f}")
            update_history_table()
            feedback_label.config(text=f"Updated: {match} - {result}\nNew Bankroll: €{bankroll:.2f}")
            logging.info(f"Updated result: {match} - {result}, New Bankroll: €{bankroll:.2f}")
        else:
            feedback_label.config(text="Pending bet mismatch.")
            logging.warning("Pending bet mismatch.")

def delete_selected_bet(selected_item):
    if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this bet?"):
        global history, bankroll
        
        # Get the index of the selected item
        item_values = tree.item(selected_item)['values']
        match_name = item_values[0]
        stake = float(item_values[2])
        result = item_values[3]
        payout = float(item_values[4])
        
        # Update bankroll if the bet was completed
        if result != "Pending":
            if result == "Win":
                bankroll -= (payout - stake)
            else:
                bankroll += stake
            save_bankroll(bankroll)
            br_label.config(text=f"Bankroll: €{bankroll:.2f}")
        
        # Remove from history DataFrame
        history = history[history["Match"] != match_name]
        save_history(history)
        update_history_table()
        feedback_label.config(text=f"Deleted bet: {match_name}")
        logging.info(f"Deleted bet: {match_name}")

def show_context_menu(event):
    selected_item = tree.selection()
    if selected_item:
        menu = tk.Menu(root, tearoff=0, bg="#2E3B55", fg="white")
        item_values = tree.item(selected_item)['values']
        result = item_values[3]
        
        if result == "Pending":
            menu.add_command(label="Mark as Win", 
                           command=lambda: update_bet_result(selected_item, "Win"),
                           background="#4CAF50", foreground="white")
            menu.add_command(label="Mark as Loss", 
                           command=lambda: update_bet_result(selected_item, "Loss"),
                           background="#F44336", foreground="white")
            menu.add_separator()
        
        menu.add_command(label="Delete", 
                        command=lambda: delete_selected_bet(selected_item),
                        background="#FF5722", foreground="white")
        menu.tk_popup(event.x_root, event.y_root)

def update_bet_result(selected_item, result):
    global history, bankroll
    item_values = tree.item(selected_item)['values']
    match_name = item_values[0]
    odds = float(item_values[1])
    stake = float(item_values[2])
    
    # Find the bet in history
    idx = history[history["Match"] == match_name].index[0]
    
    # Update result and bankroll
    if result == "Win":
        payout = stake * odds
        bankroll += payout - stake
        history.loc[idx, "Payout"] = payout
    else:
        bankroll -= stake
        history.loc[idx, "Payout"] = 0
    
    history.loc[idx, "Result"] = result
    save_bankroll(bankroll)
    save_history(history)
    br_label.config(text=f"Bankroll: €{bankroll:.2f}")
    update_history_table()
    feedback_label.config(text=f"Updated: {match_name} - {result}\nNew Bankroll: €{bankroll:.2f}")

def show_stats_page():
    # Hide all frames first
    for frame in [br_frame, upload_frame, history_frame, button_frame]:
        frame.pack_forget()
    
    stats_frame.pack(fill="both", expand=True, padx=20, pady=10)
    update_stats_display()

def show_main_page():
    # Hide stats frame
    stats_frame.pack_forget()
    
    # Show main page frames
    br_frame.pack(pady=10, fill="x")
    upload_frame.pack(pady=10)
    history_frame.pack(pady=10, fill="both", expand=True)
    data_frame.pack(pady=5, fill="x")
    button_frame.pack(pady=10)

def update_stats_display(period=None):
    if history.empty:
        messagebox.showinfo("Stats", "No betting history yet.")
        logging.info("No betting history for stats.")
        return
    
    # Clear previous stats
    for widget in stats_frame.winfo_children():
        widget.destroy()
    
    # Add back button
    back_btn = tk.Button(stats_frame, text="← Back", 
                        bg="#4CAF50", fg="white",
                        command=show_main_page)
    back_btn.pack(anchor="w", pady=(0, 20))
    
    # Convert Date column to datetime
    history['Date'] = pd.to_datetime(history['Date'], format='%d.%m.%Y')
    
    # Filter data based on period
    current_date = pd.Timestamp.now()
    if period == 'day':
        filtered_history = history[history['Date'].dt.date == current_date.date()]
        period_text = "Today's Bets"
    elif period == 'week':
        filtered_history = history[history['Date'] >= current_date - pd.Timedelta(days=7)]
        period_text = "Last 7 Days"
    elif period == 'month':
        filtered_history = history[history['Date'] >= current_date - pd.Timedelta(days=30)]
        period_text = "Last 30 Days"
    else:
        filtered_history = history
        period_text = "All Time"
    
    # Period selection frame
    period_frame = tk.Frame(stats_frame, bg="#2E3B55", padx=20, pady=10)
    period_frame.pack(fill="x", pady=(0, 20))
    
    tk.Label(period_frame, text=f"Statistics - {period_text}", 
             font=("Helvetica", 14, "bold"),
             fg="#00B4D8", bg="#2E3B55").pack(pady=10)
    
    # Calculate stats
    total_bets = len(filtered_history)
    completed_bets = len(filtered_history[filtered_history["Result"] != "Pending"])
    wins = len(filtered_history[filtered_history["Result"] == "Win"])
    losses = len(filtered_history[filtered_history["Result"] == "Loss"])
    pending = len(filtered_history[filtered_history["Result"] == "Pending"])
    
    win_rate = (wins / completed_bets * 100) if completed_bets > 0 else 0
    total_stake = filtered_history["Stake"].sum()
    total_payout = filtered_history["Payout"].sum()
    total_profit = total_payout - total_stake
    roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
    
    # Create stats sections
    sections = [
        ("Summary", [
            f"Period: {period_text}",
            f"Total Bets: {total_bets}",
            f"Completed: {completed_bets}",
            f"Pending: {pending}"
        ]),
        ("Performance", [
            f"Wins: {wins}",
            f"Losses: {losses}",
            f"Win Rate: {win_rate:.1f}%",
            f"ROI: {roi:.1f}%"
        ]),
        ("Financial", [
            f"Total Stake: €{total_stake:.2f}",
            f"Total Payout: €{total_payout:.2f}",
            f"Net Profit: €{total_profit:.2f}"
        ])
    ]
    
    # Add daily breakdown if viewing week or month
    if period in ['week', 'month']:
        daily_stats = filtered_history.groupby(filtered_history['Date'].dt.date).agg({
            'Stake': 'sum',
            'Payout': 'sum',
            'Result': lambda x: sum(x == 'Win')
        }).reset_index()
        daily_stats['Profit'] = daily_stats['Payout'] - daily_stats['Stake']
        
        daily_breakdown = ["Daily Breakdown:"]
        for _, row in daily_stats.iterrows():
            date_str = row['Date'].strftime('%d/%m/%Y')
            daily_breakdown.append(
                f"  • {date_str}: €{row['Profit']:.2f} ({row['Result']} wins)"
            )
        sections.append(("Daily Performance", daily_breakdown))
    
    # Display sections
    for title, stats in sections:
        section_frame = tk.Frame(stats_frame, bg="#2E3B55", padx=20, pady=15)
        section_frame.pack(fill="x", pady=10, padx=20)
        
        tk.Label(section_frame, text=title, font=("Helvetica", 12, "bold"),
                fg="#00B4D8", bg="#2E3B55").pack(anchor="w", pady=(0, 10))
        
        for stat in stats:
            tk.Label(section_frame, text=stat, font=("Helvetica", 11),
                    fg="#D3D3D3", bg="#2E3B55").pack(anchor="w", pady=2)

def update_history_table():
    for item in tree.get_children():
        tree.delete(item)
    
    # Configure tags for different row states
    tree.tag_configure('pending', background='#2C3E50', foreground='#FFC107')  # Pending bets in amber
    tree.tag_configure('win', background='#1B5E20', foreground='#FFFFFF')      # Wins in green
    tree.tag_configure('loss', background='#7F1D1D', foreground='#FFFFFF')     # Losses in red
    tree.tag_configure('striped', background='#1E2B45')                        # Striped rows
    
    for idx, row in history.iterrows():
        # Determine row tags
        tags = []
        if row["Result"] == "Pending":
            tags.append('pending')
        elif row["Result"] == "Win":
            tags.append('win')
        elif row["Result"] == "Loss":
            tags.append('loss')
        
        # Add striped effect for even rows
        if idx % 2 == 0:
            tags.append('striped')
        
        # Format payout with color indicators
        if row["Result"] == "Win":
            payout_str = f"+€{row['Payout']:.2f}"
        elif row["Result"] == "Loss":
            payout_str = f"-€{row['Stake']:.2f}"
        else:
            payout_str = f"€{row['Payout']:.2f}"
        
        # Insert row with appropriate tags
        item = tree.insert("", "end", values=(
            row["Match"],
            f"{row['Odds']:.2f}",
            f"€{row['Stake']:.2f}",
            row["Result"],
            payout_str,
            "✓ Win | ✗ Lose" if row["Result"] == "Pending" else ""
        ), tags=tags)

# GUI Setup
root = TkinterDnD.Tk()
logging.info("TkinterDnD root created.")
root.title("Betting Calculator")
root.geometry("900x800")
root.configure(bg="#1A2238")

# Variables
bankroll = load_bankroll()
history = load_history()

# Styles
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=5)
style.configure("TLabel", font=("Helvetica", 14), background="#1A2238", foreground="#D3D3D3")

# Configure custom style for feedback label
style.configure("Feedback.TLabel",
                font=("Helvetica", 12),
                background="#1E2B45",  # Darker background
                foreground="#FFCA28",  # Amber text
                padding=(10, 10))      # Add some padding

# Configure Treeview colors and fonts
style.configure("Treeview",
                font=("Helvetica", 10),
                background="#2E3B55",
                foreground="#FFFFFF",
                fieldbackground="#2E3B55",
                rowheight=30)  # Increased row height

style.configure("Treeview.Heading",
                font=("Helvetica", 11, "bold"),
                background="#1E2B45",
                foreground="#00B4D8",
                padding=(10, 5))  # Added padding to headers

# Configure tags for different bet states
style.map('Treeview',
          background=[('selected', '#3D4B6A')],  # Darker blue when selected
          foreground=[('selected', '#FFFFFF')])

# Create all frames
br_frame = tk.Frame(root, bg="#1A2238")
upload_frame = tk.Frame(root, bg="#1A2238")
history_frame = tk.Frame(root, bg="#1A2238")
button_frame = tk.Frame(root, bg="#1A2238")
stats_frame = tk.Frame(root, bg="#1A2238")

# Bankroll Frame setup
br_label = ttk.Label(br_frame, text=f"Bankroll: €{bankroll:.2f}", foreground="#00B4D8")
br_label.grid(row=0, column=0, columnspan=3, pady=5)

tk.Button(br_frame, text="+€50", bg="#4CAF50", fg="white", command=lambda: adjust_bankroll(50)).grid(row=1, column=0, padx=5)
tk.Button(br_frame, text="-€50", bg="#F44336", fg="white", command=lambda: adjust_bankroll(-50)).grid(row=1, column=1, padx=5)
tk.Button(br_frame, text="Reset", bg="#607D8B", fg="white", command=lambda: adjust_bankroll(200 - bankroll)).grid(row=1, column=2, padx=5)

# File Upload Area
upload_label = ttk.Label(upload_frame, text="Drag Screenshot Here (Bet or Result)", 
                        foreground="#D3D3D3", font=("Helvetica", 12))
upload_label.pack(pady=10)

# Drop area with dark theme
drop_area = tk.Label(upload_frame, 
                    text="Drop Zone", 
                    bg="#1A2238",  # Dark background matching the theme
                    fg="#D3D3D3",  # Light gray text
                    width=50, 
                    height=8, 
                    relief="solid",  # Solid border instead of sunken
                    borderwidth=2,   # Border width
                    font=("Helvetica", 10))
drop_area.pack(pady=10, padx=20)

# Setup drag-and-drop
drop_area.drop_target_register(DND_FILES)
drop_area.dnd_bind('<<Drop>>', on_drop)

# Create a frame for the feedback area with dark background
feedback_frame = tk.Frame(upload_frame, bg="#1E2B45", relief="solid", borderwidth=1)
feedback_frame.pack(pady=5, padx=20, fill="x")

# Feedback Label with dark background
feedback_label = ttk.Label(feedback_frame, 
                          text="", 
                          style="Feedback.TLabel",
                          wraplength=600)  # Allow text to wrap
feedback_label.pack(pady=10, padx=10, fill="x")

# History Table
tree = ttk.Treeview(history_frame, 
                    columns=("Match", "Odds", "Stake", "Result", "Payout", "Actions"),
                    show="headings",
                    height=15)

# Configure columns
tree.heading("Match", text="Match")
tree.heading("Odds", text="Odds")
tree.heading("Stake", text="Stake")
tree.heading("Result", text="Result")
tree.heading("Payout", text="Payout")
tree.heading("Actions", text="Actions")

# Configure column widths and alignment
tree.column("Match", width=250, anchor="w")  # Left align, wider for match names
tree.column("Odds", width=70, anchor="center")
tree.column("Stake", width=80, anchor="center")
tree.column("Result", width=80, anchor="center")
tree.column("Payout", width=100, anchor="center")
tree.column("Actions", width=120, anchor="center")

# Add some padding around the tree
tree.pack(fill="both", expand=True, padx=20, pady=10)

# Define tree-related functions after tree creation
def on_tree_click(event):
    region = tree.identify("region", event.x, event.y)
    if region == "cell":
        column = tree.identify_column(event.x)
        item = tree.identify_row(event.y)
        if column == "#6":  # Actions column
            col_value = tree.item(item)["values"][5]
            if col_value == "✓ Win | ✗ Lose":  # Only for pending bets
                # Calculate click position relative to cell width
                cell_box = tree.bbox(item, "#6")
                if cell_box:
                    relative_x = event.x - cell_box[0]
                    if relative_x < cell_box[2] / 2:  # Left half (Win)
                        update_bet_result(item, "Win")
                    else:  # Right half (Lose)
                        update_bet_result(item, "Loss")

# Bind tree events
tree.bind('<Button-1>', on_tree_click)
tree.bind("<Button-3>", show_context_menu)

# Buttons Frame
ttk.Button(button_frame, text="View Stats", command=show_stats_page).pack(side="left", padx=10)
ttk.Button(button_frame, text="Weekly Stats", 
          command=lambda: (show_stats_page(), update_stats_display('week'))).pack(side="left", padx=10)
ttk.Button(button_frame, text="Monthly Stats", 
          command=lambda: (show_stats_page(), update_stats_display('month'))).pack(side="left", padx=10)
ttk.Button(button_frame, text="Browse File", 
          command=lambda: process_file(filedialog.askopenfilename(
              filetypes=[("Image files", "*.png *.jpg *.jpeg")]))).pack(side="left", padx=10)

# Create data management frame
data_frame = tk.Frame(root, bg="#1A2238")
data_frame.pack(pady=5, fill="x")

# Style for data buttons
data_button_style = {
    "font": ("Helvetica", 11),
    "bg": "#2E3B55",
    "fg": "white",
    "width": 15,
    "height": 1,
    "relief": "flat",
    "padx": 10
}

# Data management buttons
tk.Label(data_frame, 
        text="Data Analysis:", 
        font=("Helvetica", 12, "bold"),
        bg="#1A2238",
        fg="#00B4D8").pack(side="left", padx=20)

tk.Button(data_frame,
         text="Daily Overview",
         command=lambda: (show_stats_page(), update_stats_display('day')),
         **data_button_style).pack(side="left", padx=5)

tk.Button(data_frame,
         text="Weekly Analysis",
         command=lambda: (show_stats_page(), update_stats_display('week')),
         **data_button_style).pack(side="left", padx=5)

tk.Button(data_frame,
         text="Monthly Report",
         command=lambda: (show_stats_page(), update_stats_display('month')),
         **data_button_style).pack(side="left", padx=5)

# Show initial page
show_main_page()
update_history_table()

# Start the application
root.mainloop()
logging.info("Application closed.")