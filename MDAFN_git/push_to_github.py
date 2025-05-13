#!/usr/bin/env python
import os
import subprocess
import sys
import time
from getpass import getpass

def run_command(command, verbose=True, check=True, timeout=30):
    """Run a system command and return the output."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True, check=check, timeout=timeout)
        if verbose and result.stdout:
            print(result.stdout)
        return result.stdout.strip() if result.stdout else "", result.returncode
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"Error executing command: {command}")
            print(f"Error: {e.stderr}")
        return e.stderr, e.returncode
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {command}")
        return "", 1
    except Exception as e:
        print(f"Exception running command: {str(e)}")
        return "", 1

def main():
    print("\n===== MDAFN GitHub Upload Tool =====\n")
    
    # Check if git is installed
    stdout, code = run_command("git --version", verbose=False)
    if code != 0:
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
        
        # Check if GitHub CLI is available
        stdout, code = run_command("gh --version", verbose=False, check=False)
        has_gh_cli = code == 0
        
        # GitHub CLI approach
        if has_gh_cli:
            print("\nCreating GitHub repository using GitHub CLI...")
            stdout, code = run_command(f'gh repo create {repo_name} {private_flag} --description "{repo_description}" --source=. --remote=origin --push')
            
            if code == 0:
                print(f"\n✅ Repository successfully created and code pushed to GitHub!")
                print(f"🌐 Repository URL: https://github.com/{github_username}/{repo_name}")
                return
            else:
                print("GitHub CLI encountered an error. Falling back to manual approach.")
        else:
            print("\nGitHub CLI (gh) is not installed. Using personal access token method instead.")
        
        # GitHub Personal Access Token approach
        print("\nTo create a repository and push your code, you'll need a GitHub Personal Access Token.")
        print("Please follow these steps:")
        print("1. Go to https://github.com/settings/tokens")
        print("2. Click 'Generate new token' (classic)")
        print("3. Give it a name (e.g. 'MDAFN Upload')")
        print("4. Select at least the 'repo' scope")
        print("5. Click 'Generate token' and copy the token")
        
        token = getpass("\nEnter your GitHub Personal Access Token: ")
        
        # Create the repository via API if needed
        # Note: this is optional as the push can create the repo too
        print("\nSetting up remote repository...")
        
        # Set the remote origin
        remote_url = f"https://{github_username}:{token}@github.com/{github_username}/{repo_name}.git"
        stdout, code = run_command(f'git remote add origin {remote_url}')
        
        if code != 0:
            # If remote already exists, set the URL
            stdout, code = run_command(f'git remote set-url origin {remote_url}')
        
        # Determine default branch name
        branch_name_output, _ = run_command("git branch --show-current", verbose=False)
        branch_name = branch_name_output.strip() or "main"  # Default to main if no branch exists
        
        # Push to GitHub
        print(f"\nPushing code to GitHub (branch: {branch_name})...")
        stdout, code = run_command(f"git push -u origin {branch_name}")
        
        if code == 0:
            print(f"\n✅ Repository successfully created and code pushed to GitHub!")
            print(f"🌐 Repository URL: https://github.com/{github_username}/{repo_name}")
        else:
            print("\n❌ Push failed. Please check your token permissions and try again.")
    else:
        # For existing repository
        remote_url = input("\nEnter the GitHub repository URL: ")
        run_command(f'git remote add origin {remote_url}')
        
        # Get current branch
        branch_name_output, _ = run_command("git branch --show-current", verbose=False)
        branch_name = branch_name_output.strip() or "main"  # Default to main if no branch exists
        
        stdout, code = run_command(f"git push -u origin {branch_name}")
        
        if code == 0:
            print(f"\n✅ Code successfully pushed to GitHub!")
        else:
            print("\n❌ Push failed. Please check your repository URL and permissions.")

if __name__ == "__main__":
    main() 