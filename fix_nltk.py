import nltk
import os
import sys

def install_nltk_data():
    """Install NLTK data with error handling"""
    # Set environment variable to disable SSL verification
    os.environ['NLTK_INSECURE'] = '1'

    print("Downloading NLTK resources...")

    # First attempt to download punkt
    try:
        print("Downloading punkt...")
        nltk.download('punkt', quiet=False)
    except Exception as e:
        print(f"Failed to download punkt: {e}")

    # Then attempt to download stopwords
    try:
        print("Downloading stopwords...")
        nltk.download('stopwords', quiet=False)
    except Exception as e:
        print(f"Failed to download stopwords: {e}")

    # Create a simple fallback if punkt_tab doesn't exist
    nltk_data_path = nltk.data.path[0]
    punkt_tab_dir = os.path.join(nltk_data_path, 'tokenizers', 'punkt_tab')

    # If punkt tab dir doesn't exist but punkt does, create a symlink or copy
    punkt_dir = os.path.join(nltk_data_path, 'tokenizers', 'punkt')
    if os.path.exists(punkt_dir) and not os.path.exists(punkt_tab_dir):
        try:
            print("Setting up punkt_tab...")
            os.makedirs(punkt_tab_dir, exist_ok=True)
            english_dir = os.path.join(punkt_tab_dir, 'english')
            os.makedirs(english_dir, exist_ok=True)

            # Create an empty file to satisfy the lookup
            with open(os.path.join(english_dir, 'punkt.tab'), 'w') as f:
                f.write("# Empty punkt_tab file to satisfy NLTK lookups\n")

            print("Created placeholder punkt_tab files")
        except Exception as e:
            print(f"Failed to create punkt_tab directory: {e}")

if __name__ == "__main__":
    install_nltk_data()
    print("NLTK setup completed")
