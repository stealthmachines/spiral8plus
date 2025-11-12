#!/usr/bin/env python3
"""
Git Repository Initialization Script for Human Genome Visualization Project

This script helps initialize a clean git repository with proper structure,
excluding large genome data files and setting up the initial commit.

Usage:
    python init_git_repo.py

This will:
1. Initialize git repository if not already done
2. Add all necessary files to .gitignore
3. Make initial commit with project structure
4. Provide instructions for pushing to GitHub
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Initialize git repository for the project."""
    project_root = Path(__file__).parent

    print("🚀 Initializing Human Genome Visualization Project Git Repository")
    print("=" * 70)

    # Check if already a git repo
    if (project_root / ".git").exists():
        print("⚠️  Repository already initialized. Checking status...")
        run_command("git status", "Checking git status")
        return

    # Initialize git repository
    if not run_command("git init", "Initializing git repository"):
        return

    # Configure git (optional but recommended)
    run_command('git config user.name "Human Genome Visualization Project"', "Setting default user name")
    run_command('git config user.email "genomeviz@example.com"', "Setting default user email")

    # Add .gitignore first
    if (project_root / ".gitignore").exists():
        run_command("git add .gitignore", "Adding .gitignore")
        run_command('git commit -m "Initial commit: Add .gitignore for genome data exclusion"', "Making initial commit")

    # Add core project files
    core_files = [
        "README.md",
        "LICENSE",
        "requirements.txt",
        "genome_loader.py",
        "human_genome_control_panel.py",
        ".gitattributes",
        "CONTRIBUTING.md",
        "HUMAN_SCRIPTS_CONTROL_PANEL.md",
        "HUMAN_SCRIPTS_RATINGS.md",
        "REPOSITORY_RATING.md"
    ]

    files_to_add = []
    for file in core_files:
        if (project_root / file).exists():
            files_to_add.append(file)

    if files_to_add:
        run_command(f"git add {' '.join(files_to_add)}", "Adding core project files")
        run_command('git commit -m "Add core project files and documentation"', "Committing core files")

    # Add visualization scripts (but not data files)
    script_pattern = "human_*.py"
    if list(project_root.glob(script_pattern)):
        run_command(f"git add {script_pattern}", "Adding visualization scripts")
        run_command('git commit -m "Add human genome visualization scripts"', "Committing visualization scripts")

    # Add build scripts
    build_files = ["build_*.bat", "build_*.sh", "*.c", "*.def", "*.h"]
    build_to_add = []
    for pattern in build_files:
        build_to_add.extend([str(f) for f in project_root.glob(pattern)])

    if build_to_add:
        run_command(f"git add {' '.join(build_to_add)}", "Adding build scripts and C code")
        run_command('git commit -m "Add C engine build scripts and source code"', "Committing build files")

    # Final status
    print("\n" + "=" * 70)
    print("🎉 Git repository initialized successfully!")
    print("\n📋 Next steps:")
    print("1. Create a repository on GitHub")
    print("2. Run these commands:")
    print(f"   git remote add origin https://github.com/YOUR_USERNAME/human-genome-visualization.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    print("\n📁 Repository structure:")
    run_command("git ls-files | head -20", "Showing tracked files")

    print("\n⚠️  Important reminders:")
    print("- Genome data files are excluded from git (.gitignore)")
    print("- Use 'python genome_loader.py' to download data")
    print("- Large files (>100MB) should never be committed")
    print("- Test your scripts before pushing changes")

    print("\n🔗 Useful commands:")
    print("git status              # Check current status")
    print("git log --oneline       # View commit history")
    print("python genome_loader.py --verify  # Test installation")

if __name__ == "__main__":
    main()