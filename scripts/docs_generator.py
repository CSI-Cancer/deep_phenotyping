#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil
from pathlib import Path

def create_docs():
    """Generate documentation using pdoc."""
    print("Installing pdoc if not already installed...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pdoc3"])
    
    # Define output directory and template directory
    output_dir = "docs"
    template_dir = "docs_templates"
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define modules to document with their specific template directories
    modules = [
        {"name": "src.representation_learning", 
         "template_dir": os.path.join(template_dir, "src/representation_learning")},
        {"name": "src.leukocyte_classifier", 
         "template_dir": os.path.join(template_dir, "src/leukocyte_classifier")},
        {"name": "src.utils", 
         "template_dir": os.path.join(template_dir, "src/utils")},
        {"name": "pipeline.src", 
         "template_dir": os.path.join(template_dir, "pipeline/src")}
    ]
    
    # Get the absolute path to the repository root
    repo_root = os.path.abspath(os.path.dirname(__file__))
    
    # Set up environment variables for Python path
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root
    
    # Generate documentation for each module with its specific template
    for module_info in modules:
        module_name = module_info["name"]
        module_template_dir = module_info["template_dir"]
        
        # Use the module-specific template if it exists, otherwise use the default template
        template_option = [
            "--template-dir", module_template_dir
        ] if os.path.exists(module_template_dir) else [
            "--template-dir", template_dir
        ]
        
        print(f"Generating docs for {module_name} using template in {module_template_dir}...")
        
        cmd = [
            sys.executable, "-m", "pdoc",
            "--html",
            "--output-dir", output_dir,
            "--force",
            module_name
        ] + template_option
        
        subprocess.run(cmd, env=env)
    
    # Copy the index.html template to the output directory
    index_template = os.path.join(template_dir, "index.html")
    index_output = os.path.join(output_dir, "index.html")
    
    if os.path.exists(index_template):
        shutil.copy2(index_template, index_output)
        print(f"Created main index page at {index_output}")
    else:
        print("Warning: index.html template not found, skipping main index creation")
    
    print(f"Documentation generated in {output_dir} directory")
    print(f"Open {os.path.abspath(index_output)} in your browser to view")

if __name__ == "__main__":
    create_docs()