#!/usr/bin/env python
import os
import subprocess
import sys
from getpass import getpass

def run_command(command, verbose=True):
    """Run a system command and return the output."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True, check=True)
        if verbose and result.stdout:
            print(result.stdout)
        return result.stdout.strip() if result.stdout else ""
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def main():
    print("\n===== MDAFN GitHub Upload Tool =====\n")
    
    # Check if git is installed
    try:
        run_command("git --version", verbose=False)
    except Exception:
        print("Git is not installed or not in PATH. Please install Git first.")
        sys.exit(1)
    
    # Gather required information
    github_username = input("Enter your GitHub username: ")
    repo_name = input("Enter repository name (default: MDAFN): ") or "MDAFN"
    repo_description = input("Enter repository description (optional): ") or "Mutual Distillation Attribute Fusion Network for Multimodal Vehicle Object Detection"
    
    # Check if git repository is already initialized
    if not os.path.exists(".git"):
        print("\nInitializing Git repository...")
        run_command("git init")
    
    # Configure Git credentials
    git_name = input("\nEnter your name for Git commits: ")
    git_email = input("Enter your email for Git commits: ")
    run_command(f'git config user.name "{git_name}"')
    run_command(f'git config user.email "{git_email}"')
    
    # Create .gitignore if it doesn't exist
    if not os.path.exists(".gitignore"):
        print("\nCreating .gitignore file...")
        with open(".gitignore", "w") as f:
            f.write("# Python\n__pycache__/\n*.py[cod]\n*$py.class\n*.so\n.Python\nenv/\nbuild/\ndevelop-eggs/\ndist/\ndownloads/\neggs/\n.eggs/\nlib/\nlib64/\nparts/\nsdist/\nvar/\n*.egg-info/\n.installed.cfg\n*.egg\n\n")
            f.write("# PyTorch\n*.pth\n*.pt\n\n")
            f.write("# Jupyter\n.ipynb_checkpoints\n\n")
            f.write("# IDE\n.idea/\n.vscode/\n*.swp\n*.swo\n\n")
            f.write("# OS\n.DS_Store\n.DS_Store?\n._*\n.Spotlight-V100\n.Trashes\nehthumbs.db\nThumbs.db\n")
    
    # Add all files to staging
    print("\nAdding files to Git...")
    run_command("git add .")
    
    # Commit changes
    commit_message = input("\nEnter commit message (default: Initial commit): ") or "Initial commit"
    print("\nCommitting changes...")
    run_command(f'git commit -m "{commit_message}"')
    
    # Create repository on GitHub
    create_repo = input("\nCreate new repository on GitHub? (y/n): ").lower() == 'y'
    
    if create_repo:
        is_private = input("Make repository private? (y/n): ").lower() == 'y'
        private_flag = "--private" if is_private else "--public"
        
        # Create GitHub repository using GitHub CLI if available, or guide user to create manually
        try:
            # Try using GitHub CLI (gh)
            run_command("gh --version", verbose=False)
            print("\nCreating GitHub repository using GitHub CLI...")
            
            run_command(f'gh repo create {repo_name} {private_flag} --description "{repo_description}" --source=. --remote=origin --push')
            print(f"\n✅ Repository successfully created and code pushed to GitHub!")
            print(f"🌐 Repository URL: https://github.com/{github_username}/{repo_name}")
            return
        except Exception:
            # GitHub CLI not available, continue with manual instructions
            pass
        
        # GitHub Personal Access Token approach
        print("\nTo create a repository and push your code, you'll need a GitHub Personal Access Token.")
        print("Go to https://github.com/settings/tokens and create a token with 'repo' permissions.")
        token = getpass("Enter your GitHub Personal Access Token: ")
        
        # Set the remote origin
        remote_url = f"https://{github_username}:{token}@github.com/{github_username}/{repo_name}.git"
        run_command(f'git remote add origin {remote_url}')
        
        # Push to GitHub
        print("\nPushing code to GitHub...")
        run_command("git push -u origin master")
        
        print(f"\n✅ Repository successfully created and code pushed to GitHub!")
        print(f"🌐 Repository URL: https://github.com/{github_username}/{repo_name}")
    else:
        # For existing repository
        remote_url = input("\nEnter the GitHub repository URL: ")
        run_command(f'git remote add origin {remote_url}')
        run_command("git push -u origin master")
        print(f"\n✅ Code successfully pushed to GitHub!")

if __name__ == "__main__":
    main() 