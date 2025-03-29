# Betting Calculator

A Python application for tracking sports betting history and calculating profits/losses. Features include:
- Automatic bet extraction from screenshots
- Win/Loss tracking
- Bankroll management
- Detailed statistics view
- Daily and monthly performance tracking

## Prerequisites

1. Python 3.8 or higher
2. Tesseract OCR - [Download and install from here](https://github.com/UB-Mannheim/tesseract/wiki)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/betting-calculator.git
cd betting-calculator
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Make sure Tesseract OCR is installed and the path in `betting_calculator.py` matches your installation:
```python
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Usage

1. Run the application:
```bash
python betting_calculator.py
```

2. Features:
   - Drag and drop bet screenshots to automatically extract information
   - Right-click on bets to mark as Win/Loss or delete
   - Use the Actions column to quickly mark pending bets as Win/Loss
   - View statistics by different time periods (All Time, Weekly, Monthly)
   - Manage bankroll with quick adjustment buttons

## Data Storage

The application stores data in two files:
- `bankroll.txt`: Current bankroll amount
- `betting_history.csv`: Complete betting history with results

## Contributing

Feel free to submit issues and enhancement requests! 