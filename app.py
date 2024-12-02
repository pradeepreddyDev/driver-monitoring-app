import sys
import os

# Add the 'src' directory to the system path so Python can find infer.py
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from infer import main  # Assuming main function is defined in infer.py

if __name__ == "__main__":
    print("Starting Driver Monitoring Application...")
    main()
