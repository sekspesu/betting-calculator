import tkinter as tk
from tkinter import messagebox
import pandas as pd
from PIL import Image
import pytesseract

def test_libraries():
    # Create the main window
    root = tk.Tk()
    root.title("Library Test")
    root.geometry("400x300")
    
    # Test results
    results = []
    
    # Test tkinter
    try:
        label = tk.Label(root, text="Testing Libraries...")
        label.pack(pady=20)
        results.append("✓ Tkinter is working")
    except Exception as e:
        results.append(f"✗ Tkinter error: {str(e)}")
    
    # Test pandas
    try:
        df = pd.DataFrame({'test': [1, 2, 3]})
        results.append("✓ Pandas is working")
    except Exception as e:
        results.append(f"✗ Pandas error: {str(e)}")
    
    # Test Pillow
    try:
        img = Image.new('RGB', (100, 100), color='red')
        results.append("✓ Pillow is working")
    except Exception as e:
        results.append(f"✗ Pillow error: {str(e)}")
    
    # Test pytesseract
    try:
        # Note: This will only work if Tesseract is installed on your system
        pytesseract.get_tesseract_version()
        results.append("✓ Pytesseract is working")
    except Exception as e:
        results.append(f"✗ Pytesseract error: {str(e)}")
    
    # Display results
    result_text = "\n".join(results)
    result_label = tk.Label(root, text=result_text, justify=tk.LEFT)
    result_label.pack(pady=20)
    
    # Add a close button
    close_button = tk.Button(root, text="Close", command=root.destroy)
    close_button.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    test_libraries() 