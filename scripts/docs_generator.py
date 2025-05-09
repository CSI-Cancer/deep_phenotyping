import os
import subprocess
import sys

def create_docs():
    """Generate documentation using pdoc."""
    print("Installing pdoc if not already installed...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pdoc3"])
    
    print("Generating documentation...")
    # Define output directory
    output_dir = "docs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define modules to document
    modules = [
        "src.representation_learning",
        "src.leukocyte_classifier",
        "src.utils",
        "pipeline.src",
    ]
    
    # Generate documentation
    for module in modules:
        print(f"Generating docs for {module}...")
        subprocess.run([
            sys.executable, "-m", "pdoc", 
            "--html", 
            "--output-dir", output_dir, 
            "--force", 
            module
        ])
    
    print(f"Documentation generated in {output_dir} directory")

if __name__ == "__main__":
    create_docs()