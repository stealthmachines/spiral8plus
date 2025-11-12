"""
HUMAN GENOME VISUALIZATION CONTROL PANEL
═══════════════════════════════════════════════════════════════════════════
Interactive launcher for all human genome visualization scripts
Allows user to configure:
  - Number of nucleotides to load
  - Chromosome/region to visualize
  - Which script to run
═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import subprocess
from pathlib import Path
import glob

class ControlPanel:
    def __init__(self):
        self.workspace_dir = Path(__file__).parent
        self.available_scripts = self.discover_scripts()
        self.genome_limit = None
        self.chromosome = None
        self.start_position = None
        self.selected_script = None

    def discover_scripts(self):
        """Find all human_*.py visualization scripts"""
        scripts = {
            'eco': sorted(self.workspace_dir.glob('human_eco*.py')),
            'fasta': sorted(self.workspace_dir.glob('human_fasta*.py')),
            'spiral': sorted(self.workspace_dir.glob('human_spiral*.py')),
            'advanced': sorted([
                p for p in self.workspace_dir.glob('human_*.py')
                if any(keyword in p.stem for keyword in [
                    'eight_geometries', 'cubic_scaling', 'cross_cavity', 'waterfall'
                ])
            ])
        }
        return scripts

    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """Print control panel header"""
        print("="*70)
        print(" HUMAN GENOME VISUALIZATION CONTROL PANEL")
        print("="*70)
        print()

    def print_genome_info(self):
        """Display human genome information"""
        print("Human Genome (GRCh38.p14) Information:")
        print("  Total size: ~3.1 billion nucleotides")
        print("  Chromosomes: 1-22, X, Y, MT")
        print("  Format: FASTA (.fna)")
        print()

    def select_nucleotide_limit(self):
        """Prompt user for number of nucleotides to load"""
        print("-"*70)
        print("STEP 1: Select Number of Nucleotides")
        print("-"*70)
        print()
        print("Options:")
        print("  1. Quick Preview (10,000 nucleotides) - Instant")
        print("  2. Standard View (100,000 nucleotides) - Fast (~1 second)")
        print("  3. Detailed View (500,000 nucleotides) - Medium (~5 seconds)")
        print("  4. Extended View (1,000,000 nucleotides) - Slow (~10 seconds)")
        print("  5. Chromosome View (10,000,000 nucleotides) - Very slow (~1 minute)")
        print("  6. Full Genome (3,100,000,000 nucleotides) - VERY SLOW (not recommended)")
        print("  7. Custom amount")
        print()

        while True:
            choice = input("Select option (1-7): ").strip()

            if choice == '1':
                self.genome_limit = 10000
                break
            elif choice == '2':
                self.genome_limit = 100000
                break
            elif choice == '3':
                self.genome_limit = 500000
                break
            elif choice == '4':
                self.genome_limit = 1000000
                break
            elif choice == '5':
                self.genome_limit = 10000000
                break
            elif choice == '6':
                confirm = input("⚠️  WARNING: This will load 3.1GB! Continue? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    self.genome_limit = 'all'
                    break
                else:
                    print("Cancelled. Please choose another option.\n")
                    continue
            elif choice == '7':
                try:
                    custom = input("Enter number of nucleotides: ").strip()
                    self.genome_limit = int(custom)
                    if self.genome_limit < 1:
                        print("❌ Must be at least 1 nucleotide\n")
                        continue
                    break
                except ValueError:
                    print("❌ Invalid number. Please try again.\n")
                    continue
            else:
                print("❌ Invalid choice. Please enter 1-7.\n")
                continue

        print(f"\n✓ Selected: {self.genome_limit:,} nucleotides" if isinstance(self.genome_limit, int) else "\n✓ Selected: Full genome")
        print()

    def select_chromosome(self):
        """Prompt user for chromosome/region selection"""
        print("-"*70)
        print("STEP 2: Select Chromosome/Region")
        print("-"*70)
        print()
        print("Options:")
        print("  1. Chromosome 1 (NC_000001.11) - Largest, 249M bases")
        print("  2. Chromosome 2 (NC_000002.12) - 242M bases")
        print("  3. Chromosome X (NC_000023.11) - 154M bases")
        print("  4. Chromosome Y (NC_000024.10) - 57M bases")
        print("  5. Mitochondrial DNA (NC_012920.1) - 16,569 bases")
        print("  6. Start from beginning (default)")
        print("  7. Custom chromosome by ID")
        print("  8. Custom start position")
        print()

        while True:
            choice = input("Select option (1-8): ").strip()

            if choice == '1':
                self.chromosome = 'NC_000001.11'
                self.start_position = 0
                break
            elif choice == '2':
                self.chromosome = 'NC_000002.12'
                self.start_position = 0
                break
            elif choice == '3':
                self.chromosome = 'NC_000023.11'
                self.start_position = 0
                break
            elif choice == '4':
                self.chromosome = 'NC_000024.10'
                self.start_position = 0
                break
            elif choice == '5':
                self.chromosome = 'NC_012920.1'
                self.start_position = 0
                break
            elif choice == '6':
                self.chromosome = None
                self.start_position = 0
                break
            elif choice == '7':
                custom = input("Enter chromosome ID (e.g., NC_000001.11): ").strip()
                if custom:
                    self.chromosome = custom
                    self.start_position = 0
                    break
                else:
                    print("❌ Invalid chromosome ID\n")
                    continue
            elif choice == '8':
                try:
                    pos = input("Enter start position (nucleotide number): ").strip()
                    self.start_position = int(pos)
                    if self.start_position < 0:
                        print("❌ Position must be >= 0\n")
                        continue
                    self.chromosome = None
                    break
                except ValueError:
                    print("❌ Invalid number\n")
                    continue
            else:
                print("❌ Invalid choice. Please enter 1-8.\n")
                continue

        if self.chromosome:
            print(f"\n✓ Selected: {self.chromosome}")
        elif self.start_position > 0:
            print(f"\n✓ Selected: Start at position {self.start_position:,}")
        else:
            print(f"\n✓ Selected: Start from beginning")
        print()

    def select_script(self):
        """Prompt user to select which visualization script to run"""
        print("-"*70)
        print("STEP 3: Select Visualization Script")
        print("-"*70)
        print()

        # Count scripts
        total_scripts = sum(len(scripts) for scripts in self.available_scripts.values())

        if total_scripts == 0:
            print("❌ No human_*.py scripts found in workspace!")
            return False

        print(f"Found {total_scripts} visualization scripts:\n")

        print("✅ Most scripts (68/80) now support GENOME_LIMIT/GENOME_CHROMOSOME/GENOME_START")
        print("⚠️  Some special variants (C engine, GPU) may not support environment variables\n")        # Build menu
        menu_items = []
        item_num = 1

        # Eco scripts
        if self.available_scripts['eco']:
            print(f"ECO-BASED VISUALIZATIONS ({len(self.available_scripts['eco'])} scripts):")
            for script in self.available_scripts['eco'][:10]:  # Show first 10
                print(f"  {item_num}. {script.name}")
                menu_items.append(script)
                item_num += 1
            if len(self.available_scripts['eco']) > 10:
                print(f"  ... and {len(self.available_scripts['eco']) - 10} more (type 'eco' to see all)")
            print()

        # Fasta scripts
        if self.available_scripts['fasta']:
            print(f"FASTA-BASED VISUALIZATIONS ({len(self.available_scripts['fasta'])} scripts):")
            for script in self.available_scripts['fasta'][:10]:  # Show first 10
                print(f"  {item_num}. {script.name}")
                menu_items.append(script)
                item_num += 1
            if len(self.available_scripts['fasta']) > 10:
                print(f"  ... and {len(self.available_scripts['fasta']) - 10} more (type 'fasta' to see all)")
            print()

        # Spiral scripts
        if self.available_scripts['spiral']:
            print(f"SPIRAL VISUALIZATIONS ({len(self.available_scripts['spiral'])} scripts):")
            for script in self.available_scripts['spiral']:
                print(f"  {item_num}. {script.name}")
                menu_items.append(script)
                item_num += 1
            print()

        # Advanced analysis scripts
        if self.available_scripts.get('advanced'):
            print(f"🌟 ADVANCED MATHEMATICAL ANALYSIS ({len(self.available_scripts['advanced'])} scripts):")
            for script in self.available_scripts['advanced']:
                print(f"  {item_num}. {script.name}")
                menu_items.append(script)
                item_num += 1
            print()

        # Special options
        print("SPECIAL OPTIONS:")
        print(f"  Type 'eco' - Show all eco scripts")
        print(f"  Type 'fasta' - Show all fasta scripts")
        print(f"  Type 'advanced' - Show all advanced analysis scripts")
        print(f"  Type 'list' - Show full list of all scripts")
        print()

        while True:
            choice = input(f"Select script (1-{len(menu_items)}) or command: ").strip().lower()

            if choice == 'eco':
                result = self._show_category('eco')
                if result is not None:
                    return True
                return self.select_script()
            elif choice == 'fasta':
                result = self._show_category('fasta')
                if result is not None:
                    return True
                return self.select_script()
            elif choice == 'advanced':
                result = self._show_category('advanced')
                if result is not None:
                    return True
                return self.select_script()
            elif choice == 'list':
                result = self._show_all_scripts()
                if result is not None:
                    return True
                return self.select_script()
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(menu_items):
                        self.selected_script = menu_items[idx]
                        print(f"\n✓ Selected: {self.selected_script.name}")
                        print()
                        return True
                    else:
                        print(f"❌ Please enter a number between 1 and {len(menu_items)}\n")
                except ValueError:
                    print("❌ Invalid input. Enter a number or command.\n")

    def _show_category(self, category):
        """Show all scripts in a category"""
        self.clear_screen()
        self.print_header()
        print(f"{category.upper()} SCRIPTS ({len(self.available_scripts[category])} total):\n")

        menu_items = []
        for idx, script in enumerate(self.available_scripts[category], 1):
            print(f"  {idx}. {script.name}")
            menu_items.append(script)
        print()

        while True:
            choice = input(f"Select script (1-{len(menu_items)}) or 'back': ").strip().lower()

            if choice == 'back':
                return None

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(menu_items):
                    self.selected_script = menu_items[idx]
                    print(f"\n✓ Selected: {self.selected_script.name}\n")
                    return self.selected_script
                else:
                    print(f"❌ Please enter 1-{len(menu_items)}\n")
            except ValueError:
                print("❌ Invalid input\n")

    def _show_all_scripts(self):
        """Show all available scripts"""
        self.clear_screen()
        self.print_header()

        all_scripts = []
        for category in ['eco', 'fasta', 'spiral']:
            all_scripts.extend(self.available_scripts[category])

        print(f"ALL SCRIPTS ({len(all_scripts)} total):\n")
        for idx, script in enumerate(all_scripts, 1):
            print(f"  {idx}. {script.name}")
        print()

        while True:
            choice = input(f"Select script (1-{len(all_scripts)}) or 'back': ").strip().lower()

            if choice == 'back':
                return None

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(all_scripts):
                    self.selected_script = all_scripts[idx]
                    print(f"\n✓ Selected: {self.selected_script.name}\n")
                    return self.selected_script
                else:
                    print(f"❌ Please enter 1-{len(all_scripts)}\n")
            except ValueError:
                print("❌ Invalid input\n")

    def show_summary(self):
        """Display configuration summary"""
        print("-"*70)
        print("CONFIGURATION SUMMARY")
        print("-"*70)
        print()
        print(f"Script: {self.selected_script.name}")
        print(f"Nucleotides: {self.genome_limit:,}" if isinstance(self.genome_limit, int) else "Nucleotides: Full genome")

        if self.chromosome:
            print(f"Chromosome: {self.chromosome}")
        elif self.start_position > 0:
            print(f"Start position: {self.start_position:,}")
        else:
            print(f"Region: From beginning")

        print()
        print("Environment variables that will be set:")
        print(f"  GENOME_LIMIT = {self.genome_limit}")
        if self.chromosome:
            print(f"  GENOME_CHROMOSOME = {self.chromosome}")
        if self.start_position:
            print(f"  GENOME_START = {self.start_position}")
        print()

    def launch_script(self):
        """Launch the selected script with configured parameters"""
        print("-"*70)
        print("LAUNCHING VISUALIZATION...")
        print("-"*70)
        print()

        # Set environment variables
        env = os.environ.copy()
        env['GENOME_LIMIT'] = str(self.genome_limit)

        if self.chromosome:
            env['GENOME_CHROMOSOME'] = self.chromosome

        if self.start_position:
            env['GENOME_START'] = str(self.start_position)

        # Print command for user reference (Windows PowerShell syntax)
        if os.name == 'nt':  # Windows
            env_vars = f'$env:GENOME_LIMIT="{self.genome_limit}"'
            if self.chromosome:
                env_vars += f'; $env:GENOME_CHROMOSOME="{self.chromosome}"'
            if self.start_position:
                env_vars += f'; $env:GENOME_START="{self.start_position}"'
            print(f"Command: {env_vars}; python {self.selected_script.name}")
        else:  # Linux/Mac
            env_vars = f"GENOME_LIMIT={self.genome_limit}"
            if self.chromosome:
                env_vars += f" GENOME_CHROMOSOME={self.chromosome}"
            if self.start_position:
                env_vars += f" GENOME_START={self.start_position}"
            print(f"Command: {env_vars} python {self.selected_script.name}")
        print()
        print("Press Ctrl+C to stop the visualization")
        print()

        try:
            # Launch the script
            result = subprocess.run(
                [sys.executable, str(self.selected_script)],
                env=env,
                cwd=self.workspace_dir
            )

            if result.returncode == 0:
                print("\n✓ Script completed successfully")
                return True
            else:
                print(f"\n⚠️  Script exited with code {result.returncode}")
                return False

        except KeyboardInterrupt:
            print("\n\n⚠️  Visualization stopped by user")
            return False
        except Exception as e:
            print(f"\n❌ Error launching script: {e}")
            return False

    def run(self):
        """Main control panel flow"""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_genome_info()

            # Step 1: Select nucleotides
            self.select_nucleotide_limit()

            # Step 2: Select chromosome/region
            self.select_chromosome()

            # Step 3: Select script
            if not self.select_script():
                print("❌ No script selected. Exiting.")
                return

            # Show summary
            self.show_summary()

            # Confirm launch
            confirm = input("Launch visualization? (yes/no): ").strip().lower()
            if confirm != 'yes':
                retry = input("Start over? (yes/no): ").strip().lower()
                if retry != 'yes':
                    print("\nExiting control panel.")
                    return
                continue

            # Launch
            self.launch_script()

            # Ask to run again
            print()
            again = input("Run another visualization? (yes/no): ").strip().lower()
            if again != 'yes':
                print("\nThank you for using the Human Genome Visualization Control Panel!")
                return

def main():
    """Entry point"""
    try:
        panel = ControlPanel()
        panel.run()
    except KeyboardInterrupt:
        print("\n\nControl panel interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
