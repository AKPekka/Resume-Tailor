import subprocess
import sys
import os
import platform

def check_python_version():
    """Ensure compatible Python version"""
    major, minor = sys.version_info[:2]
    if major != 3 or minor < 7:
        print(f"Error: Python 3.7+ required. You're using {sys.version}")
        sys.exit(1)

def install_package(package):
    """Install a single package with error handling"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}. Error: {e}")
        return False

def install_requirements():
    """Install required packages one by one for better error handling"""
    print("Installing required packages...")

    # Upgrade pip first
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--upgrade", "pip", "setuptools", "wheel"
        ])
    except subprocess.CalledProcessError as e:
        print(f"Failed to upgrade pip. Error: {e}")
        print("Continuing with installation...")

    # List of required packages
    packages = [
        "nltk==3.8.1",
        "PyPDF2==3.0.1",
        "python-docx==0.8.11",
        "numpy>=2.0.0,<2.3.0",  # More flexible version range
        "scikit-learn==1.4.2",
        "streamlit==1.28.0",
        "certifi==2024.2.2"
    ]

    # Install each package separately
    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)

    if failed_packages:
        print("\nThe following packages failed to install:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nYou might need to install them manually.")
        return False

    return True

def download_nltk_resources():
    """Download NLTK resources with SSL workaround"""
    print("Downloading NLTK resources...")
    try:
        import nltk
        # Temporarily disable SSL verification
        os.environ['NLTK_INSECURE'] = '1'

        resources = ['punkt', 'stopwords']
        for resource in resources:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=False)

        # Also try to download punkt_tab
        try:
            nltk.download('punkt_tab', quiet=False)
        except:
            print("punkt_tab download failed, but we'll try to continue")

    except Exception as e:
        print(f"Failed to download NLTK resources: {e}")
        print("You can manually download them using:")
        print("  python -m nltk.downloader punkt stopwords")
        return False

    return True

def verify_installations():
    """Verify critical packages are installed"""
    required = {
        'nltk': None,
        'sklearn': None,
        'streamlit': None,
        'PyPDF2': None,
        'numpy': None
    }

    print("\nVerifying installations...")
    all_installed = True
    for pkg in required:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {pkg} {version} installed")
        except ImportError:
            print(f"✗ {pkg} not installed")
            all_installed = False

    return all_installed

def setup():
    """Main setup function"""
    try:
        check_python_version()
        if not install_requirements():
            print("\n⚠️ Some packages failed to install, but we'll try to continue")

        if not download_nltk_resources():
            print("\n⚠️ NLTK resource download had issues, but we'll try to continue")

        if not verify_installations():
            print("\n⚠️ Some packages could not be verified, but we'll try to continue")
        else:
            print("\n✅ All required packages installed!")

        print("\nTo run the application:")
        print("  streamlit run main.py")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup()
