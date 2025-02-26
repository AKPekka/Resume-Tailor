import streamlit as st
import os
import sys

# Add the current directory to the system path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the streamlit app module
from streamlit_ui import main

# Run the Streamlit application
if __name__ == "__main__":
    main()
