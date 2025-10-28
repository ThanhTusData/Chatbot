import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(description='Chatbot ML/NLP Application')
    parser.add_argument('command', choices=['serve-api', 'serve-web', 'train', 'build-kb', 'streamlit', 'desktop'])
    parser.add_argument('--config', type=str, help='Config file path')
    
    args = parser.parse_args()
    
    if args.command == 'serve-api':
        from serving.fastapi_app import main as serve_api
        serve_api()
    
    elif args.command == 'serve-web':
        from web.flask_app import main as serve_web
        serve_web()
    
    elif args.command == 'train':
        from training.train_intent import main as train
        train()
    
    elif args.command == 'build-kb':
        from scripts.build_kb_index import main as build_kb
        build_kb()
    
    elif args.command == 'streamlit':
        import subprocess
        subprocess.run(['streamlit', 'run', 'src/streamlit_app.py'])
    
    elif args.command == 'desktop':
        from desktop_app import main as desktop
        desktop()

if __name__ == '__main__':
    main()