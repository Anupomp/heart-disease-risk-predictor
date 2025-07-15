#!/usr/bin/env python3
"""
Heart Disease Predictor - Application Launcher
Provides multiple ways to run the heart disease prediction system.
"""

import subprocess
import sys
import os
import argparse
import time

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['flask', 'pandas', 'scikit-learn', 'numpy', 'joblib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing dependencies: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("✅ Dependencies installed successfully!")
    else:
        print("✅ All dependencies satisfied!")

def run_api():
    """Run the heart disease prediction API."""
    print("🚀 Starting Heart Disease Prediction API...")
    print("📡 API will be available at: http://localhost:5000")
    print("📋 Health check: http://localhost:5000/health")
    print("📝 Example endpoint: http://localhost:5000/example")
    
    try:
        subprocess.run([sys.executable, 'heart_disease_api.py'])
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")

def run_web_interface():
    """Run the web interface."""
    print("🌐 Starting Heart Disease Prediction Web Interface...")
    print("🔗 Web interface will be available at: http://localhost:8080")
    
    try:
        subprocess.run([sys.executable, 'web_interface.py'])
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped")

def run_full_analysis():
    """Run the comprehensive analysis notebook."""
    print("📊 Starting Comprehensive Analysis...")
    print("🔬 This will run the complete ML pipeline and analysis")
    
    try:
        subprocess.run([sys.executable, '-m', 'jupyter', 'notebook', 'comprehensive_analysis.ipynb'])
    except KeyboardInterrupt:
        print("\n🛑 Analysis stopped")

def run_enhanced_ml():
    """Run the enhanced ML pipeline."""
    print("🤖 Starting Enhanced ML Pipeline...")
    print("⚡ This will train 11 different ML models with ensemble methods")
    
    try:
        subprocess.run([sys.executable, 'enhanced_model_pipeline.py'])
    except KeyboardInterrupt:
        print("\n🛑 ML pipeline stopped")

def run_eda():
    """Run the comprehensive EDA."""
    print("📈 Starting Comprehensive Exploratory Data Analysis...")
    print("📊 This will generate interactive visualizations and statistical analysis")
    
    try:
        subprocess.run([sys.executable, 'comprehensive_eda.py'])
    except KeyboardInterrupt:
        print("\n🛑 EDA analysis stopped")

def run_docker():
    """Run the application using Docker."""
    print("🐳 Starting Docker containers...")
    
    if not os.path.exists('outputs/docker-compose.yml'):
        print("❌ Docker configuration not found!")
        return
    
    try:
        subprocess.run(['docker-compose', '-f', 'outputs/docker-compose.yml', 'up'], cwd='.')
    except KeyboardInterrupt:
        print("\n🛑 Docker containers stopped")

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="Heart Disease Predictor - Application Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py --api                 # Run API server
  python launcher.py --web                 # Run web interface
  python launcher.py --analysis            # Run Jupyter analysis
  python launcher.py --ml                  # Run enhanced ML pipeline
  python launcher.py --eda                 # Run comprehensive EDA
  python launcher.py --docker              # Run with Docker
  python launcher.py --all                 # Run API + Web interface
        """
    )
    
    parser.add_argument('--api', action='store_true', 
                       help='Run the prediction API server')
    parser.add_argument('--web', action='store_true', 
                       help='Run the web interface')
    parser.add_argument('--analysis', action='store_true', 
                       help='Run comprehensive analysis notebook')
    parser.add_argument('--ml', action='store_true', 
                       help='Run enhanced ML pipeline')
    parser.add_argument('--eda', action='store_true', 
                       help='Run comprehensive EDA')
    parser.add_argument('--docker', action='store_true', 
                       help='Run using Docker')
    parser.add_argument('--all', action='store_true', 
                       help='Run API and web interface together')
    parser.add_argument('--skip-deps', action='store_true', 
                       help='Skip dependency check')
    
    args = parser.parse_args()
    
    # Header
    print("="*70)
    print("🏥 HEART DISEASE RISK PREDICTOR - LAUNCHER")
    print("="*70)
    
    # Check dependencies (unless skipped)
    if not args.skip_deps:
        check_dependencies()
        print()
    
    # Determine what to run
    if args.api:
        run_api()
    elif args.web:
        run_web_interface()
    elif args.analysis:
        run_full_analysis()
    elif args.ml:
        run_enhanced_ml()
    elif args.eda:
        run_eda()
    elif args.docker:
        run_docker()
    elif args.all:
        print("🚀 Starting both API and Web Interface...")
        print("📡 API: http://localhost:5000")
        print("🌐 Web: http://localhost:8080")
        
        # Start API in background
        import threading
        api_thread = threading.Thread(target=run_api)
        api_thread.daemon = True
        api_thread.start()
        
        # Wait a moment for API to start
        time.sleep(2)
        
        # Start web interface
        run_web_interface()
    else:
        # Interactive menu
        print("🎯 Choose an option:")
        print("1. 📡 Run Prediction API (REST endpoints)")
        print("2. 🌐 Run Web Interface (User-friendly UI)")
        print("3. 📊 Run Comprehensive Analysis (Jupyter Notebook)")
        print("4. 🤖 Run Enhanced ML Pipeline (11 algorithms)")
        print("5. 📈 Run Comprehensive EDA (Interactive visualizations)")
        print("6. 🐳 Run with Docker (Containerized deployment)")
        print("7. 🚀 Run Both API + Web Interface")
        print("8. ❌ Exit")
        
        while True:
            try:
                choice = input("\n👉 Enter your choice (1-8): ").strip()
                
                if choice == '1':
                    run_api()
                    break
                elif choice == '2':
                    run_web_interface()
                    break
                elif choice == '3':
                    run_full_analysis()
                    break
                elif choice == '4':
                    run_enhanced_ml()
                    break
                elif choice == '5':
                    run_eda()
                    break
                elif choice == '6':
                    run_docker()
                    break
                elif choice == '7':
                    print("🚀 Starting both API and Web Interface...")
                    # Similar to --all option
                    import threading
                    api_thread = threading.Thread(target=run_api)
                    api_thread.daemon = True
                    api_thread.start()
                    time.sleep(2)
                    run_web_interface()
                    break
                elif choice == '8':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-8.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

if __name__ == '__main__':
    main()
