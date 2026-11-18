import os

def generate_tree(startpath):
    # Folders to ignore so the output doesn't crash from 100,000 files
    # Change this to [] if you literally want EVERYTHING including .git
    EXCLUDE_DIRS = {'.git', 'node_modules', '.venv', 'venv', '__pycache__', '.next'}
    
    print(f"📁 {os.path.basename(os.path.abspath(startpath))}/")
    
    for root, dirs, files in os.walk(startpath):
        # Apply the ignore list
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        # Calculate depth for indentation
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * (level)
        print(f"{indent}├── {os.path.basename(root)}/")
        
        # Print all files in this directory
        subindent = '│   ' * (level + 1)
        for f in files:
            # Get file size
            try:
                size_bytes = os.path.getsize(os.path.join(root, f))
                if size_bytes > 1024 * 1024: # 1MB+
                    size_str = f" ({(size_bytes / (1024*1024)):.1f} MB)"
                elif size_bytes > 1024:
                    size_str = f" ({(size_bytes / 1024):.0f} KB)"
                else:
                    size_str = f" ({size_bytes} bytes)"
            except OSError:
                size_str = ""
                
            print(f"{subindent}└── {f}{size_str}")

if __name__ == "__main__":
    # Runs from the current directory
    generate_tree(".")