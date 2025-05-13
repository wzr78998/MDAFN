#!/usr/bin/env python
import os
import subprocess
import sys
import time
import re
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

def is_ssh_url(url):
    """Check if the URL is an SSH URL."""
    return url.startswith("git@") or ":" in url and "@" in url

def convert_ssh_to_https(ssh_url):
    """Convert an SSH URL to HTTPS format."""
    # Pattern: git@github.com:username/repo.git -> https://github.com/username/repo.git
    match = re.match(r"git@([^:]+):([^/]+)/(.+)", ssh_url)
    if match:
        domain, username, repo = match.groups()
        return f"https://{domain}/{username}/{repo}"
    return ssh_url

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
        
        # For new repos, set default branch to main
        run_command("git checkout -b main")
    
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
    
    # Determine branch name
    branch_name_output, _ = run_command("git branch --show-current", verbose=False)
    current_branch = branch_name_output.strip() or "main"
    
    # Ask user which branch to use, default to main
    print(f"\nCurrent branch: {current_branch}")
    use_branch = input(f"Enter branch name to push (default: main): ") or "main"
    
    # If user wants to use a different branch, create and switch to it
    if use_branch != current_branch:
        print(f"\nSwitching to branch '{use_branch}'...")
        stdout, code = run_command(f"git checkout -b {use_branch}")
        if code != 0:
            # Branch might already exist
            stdout, code = run_command(f"git checkout {use_branch}")
            if code != 0:
                print(f"Error creating or switching to branch '{use_branch}'. Using current branch '{current_branch}'.")
                use_branch = current_branch
    
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
        
        # Push to GitHub
        print(f"\nPushing code to GitHub (branch: {use_branch})...")
        stdout, code = run_command(f"git push -u origin {use_branch}")
        
        if code == 0:
            print(f"\n✅ Repository successfully created and code pushed to GitHub!")
            print(f"🌐 Repository URL: https://github.com/{github_username}/{repo_name}")
        else:
            print("\n❌ Push failed. Please check your token permissions and try again.")
    else:
        # For existing repository
        print("\nEnter the GitHub repository URL:")
        print("Note: For HTTPS format use: https://github.com/username/repo.git")
        
        remote_url = input("> ")
        
        # Check if SSH URL and warn user
        if is_ssh_url(remote_url):
            print("\n⚠️ You provided an SSH URL, which requires SSH keys to be configured.")
            print("Options:")
            print("1. Continue with SSH URL (will work only if you have SSH keys set up)")
            print("2. Switch to HTTPS URL (recommended)")
            print("3. Setup SSH keys (more advanced)")
            
            choice = input("\nSelect option [2]: ") or "2"
            
            if choice == "2":
                # Convert to HTTPS URL
                https_url = convert_ssh_to_https(remote_url)
                print(f"\nConverting to HTTPS URL: {https_url}")
                print("Using personal access token for authentication...")
                
                token = getpass("Enter your GitHub Personal Access Token: ")
                # Format for git: https://username:token@github.com/username/repo.git
                user_match = re.search(r"github\.com[/:]([^/]+)/", https_url)
                if user_match:
                    repo_owner = user_match.group(1)
                    auth_url = https_url.replace("https://", f"https://{github_username}:{token}@")
                    remote_url = auth_url
                else:
                    print("Could not parse username from URL. Using as-is.")
            elif choice == "3":
                print("\nTo set up SSH keys, follow these steps:")
                print("1. Check if you already have SSH keys: ls -la ~/.ssh")
                print("2. If not, create a new key: ssh-keygen -t ed25519 -C 'your_email@example.com'")
                print("3. Start the ssh-agent: eval '$(ssh-agent -s)'")
                print("4. Add your key: ssh-add ~/.ssh/id_ed25519")
                print("5. Copy your public key: clip < ~/.ssh/id_ed25519.pub (Windows) or pbcopy < ~/.ssh/id_ed25519.pub (Mac)")
                print("6. Add this key to your GitHub account: https://github.com/settings/keys")
                print("\nTry running this script again after setting up SSH keys.")
                return
        
        # Set the remote
        stdout, code = run_command(f'git remote add origin {remote_url}')
        
        if code != 0:
            # If remote already exists, update it
            stdout, code = run_command(f'git remote set-url origin {remote_url}')
        
        print(f"\nPushing code to GitHub (branch: {use_branch})...")
        stdout, code = run_command(f"git push -u origin {use_branch}")
        
        if code == 0:
            print(f"\n✅ Code successfully pushed to GitHub!")
        else:
            print("\n❌ Push failed. Please check your repository URL and permissions.")
            print("\nIf you're using an SSH URL and getting 'Permission denied (publickey)' errors:")
            print("1. Either set up SSH keys (see GitHub docs)")
            print("2. Or use HTTPS URL with a personal access token instead")
            print("   Run this script again and select option 2 when prompted.")

if __name__ == "__main__":
    main() 