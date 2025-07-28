#!/usr/bin/env python3
"""
Script to install Tesseract OCR for better OCR fallback support.
"""

import os
import sys
import subprocess
import platform

def install_tesseract():
    """Install Tesseract OCR based on the operating system"""
    system = platform.system().lower()
    
    print("🔧 Installing Tesseract OCR...")
    
    try:
        if system == "windows":
            print("📝 For Windows, please install Tesseract manually:")
            print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("2. Install the .exe file")
            print("3. Add Tesseract to your PATH environment variable")
            print("4. Typical installation path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
            
        elif system == "darwin":  # macOS
            print("🍎 Installing Tesseract on macOS using Homebrew...")
            subprocess.run(["brew", "install", "tesseract"], check=True)
            
        elif system == "linux":
            print("🐧 Installing Tesseract on Linux...")
            # Try different package managers
            try:
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "tesseract-ocr"], check=True)
            except subprocess.CalledProcessError:
                try:
                    subprocess.run(["sudo", "yum", "install", "-y", "tesseract"], check=True)
                except subprocess.CalledProcessError:
                    print("❌ Could not install via apt-get or yum. Please install manually.")
                    return False
        
        # Test installation
        try:
            result = subprocess.run(["tesseract", "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Tesseract installed successfully!")
            print(f"Version: {result.stdout.split()[1] if result.stdout else 'Unknown'}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Tesseract installation may have failed or is not in PATH")
            return False
            
    except Exception as e:
        print(f"❌ Error installing Tesseract: {e}")
        return False

def main():
    """Main installation function"""
    print("MTG Card Identifier - Tesseract OCR Installation")
    print("=" * 50)
    
    # Check if tesseract is already installed
    try:
        result = subprocess.run(["tesseract", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Tesseract is already installed!")
        print(f"Version: {result.stdout.split()[1] if result.stdout else 'Unknown'}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    success = install_tesseract()
    
    if success:
        print("\n🎉 Installation completed successfully!")
        print("You can now run the MTG card identifier with improved OCR support.")
    else:
        print("\n⚠️  Installation incomplete.")
        print("The system will still work with EasyOCR only, but Tesseract provides better fallback support.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)