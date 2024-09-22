import sys
import subprocess

def install_packages():
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_scripts():
    print("Running project...")
    subprocess.check_call([sys.executable, "run_pyscripts.py"])

def main():
    try:
        install_packages()
        run_scripts()
        print("Installation and setup completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()







