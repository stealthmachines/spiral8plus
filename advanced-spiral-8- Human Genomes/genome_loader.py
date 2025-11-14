#!/usr/bin/env python3
"""
Genome Data Loader for Human Genome Visualization Project

This script provides automated downloading and setup of genome datasets
from NCBI for use with the human genome visualization scripts.

Supported data sources:
- NCBI Datasets (human genome reference)
- NCBI RefSeq (bacterial genomes like E. coli)
- Custom FASTA files

Usage:
    python genome_loader.py --help
    python genome_loader.py --human-reference
    python genome_loader.py --ecoli
    python genome_loader.py --custom https://example.com/genome.fasta

Author: Human Genome Visualization Project
Date: November 12, 2025
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, List
import time

class GenomeLoader:
    """Handles downloading and setup of genome datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # NCBI Datasets API endpoints
        self.ncbi_base = "https://api.ncbi.nlm.nih.gov/datasets/v2"

        # Common genome accessions
        self.genomes = {
            'human': {
                'name': 'Human Genome Reference (GRCh38)',
                'accession': 'GCF_000001405.40',
                'description': 'Latest human genome reference assembly',
                'chromosomes': ['NC_000001.11', 'NC_000002.12', 'NC_000003.12', 'NC_000004.12', 'NC_000005.10']
            },
            'ecoli': {
                'name': 'Escherichia coli K-12 MG1655',
                'accession': 'GCF_000005845.2',
                'description': 'Well-studied laboratory strain of E. coli',
                'chromosomes': ['NC_000913.3']
            },
            'yeast': {
                'name': 'Saccharomyces cerevisiae S288C',
                'accession': 'GCF_000146045.2',
                'description': 'Budding yeast reference genome',
                'chromosomes': ['NC_001133.9', 'NC_001134.8', 'NC_001135.5', 'NC_001136.10']
            }
        }

    def download_with_progress(self, url: str, filename: str, description: str = "") -> bool:
        """Download a file with progress indication."""
        try:
            print(f"Downloading {description}...")
            print(f"From: {url}")
            print(f"To: {filename}")

            # Use curl for better progress indication
            cmd = [
                'curl', '-L', '-o', str(filename), '--progress-bar', url
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                size = os.path.getsize(filename) / (1024 * 1024)  # MB
                print(".1f"                return True
            else:
                print(f"Download failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

    def download_human_reference(self) -> bool:
        """Download human genome reference from NCBI Datasets."""
        print("=== Downloading Human Genome Reference (GRCh38) ===")

        # Use NCBI datasets command line tool if available
        try:
            # Check if datasets tool is available
            result = subprocess.run(['datasets', '--version'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                print("Using NCBI datasets command line tool...")
                return self._download_with_datasets_tool('human')
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fallback to manual download
        print("Using manual download method...")
        return self._download_human_manual()

    def _download_with_datasets_tool(self, genome_key: str) -> bool:
        """Download using NCBI datasets CLI tool."""
        genome_info = self.genomes[genome_key]
        accession = genome_info['accession']

        # Download genome data
        cmd = [
            'datasets', 'download', 'genome', 'accession', accession,
            '--include', 'genome', '--filename', str(self.data_dir / f"{genome_key}_genome.zip")
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=self.data_dir)

        if result.returncode == 0:
            # Unzip the downloaded data
            zip_file = self.data_dir / f"{genome_key}_genome.zip"
            if zip_file.exists():
                print("Extracting downloaded data...")
                import zipfile
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir / genome_key)

                # Find FASTA file
                fasta_files = list((self.data_dir / genome_key).glob("**/*.fasta"))
                if fasta_files:
                    target_fasta = self.data_dir / f"{genome_key}.fasta"
                    fasta_files[0].rename(target_fasta)
                    print(f"Genome saved as: {target_fasta}")
                    return True

        return False

    def _download_human_manual(self) -> bool:
        """Manual download of human genome reference."""
        # Use Ensembl or NCBI FTP for direct FASTA download
        base_url = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/reference/GCF_000001405.40_GRCh38.p14"

        # Try to download chromosome by chromosome for better progress
        success_count = 0
        total_chromosomes = len(self.genomes['human']['chromosomes'])

        for chrom in self.genomes['human']['chromosomes']:
            fasta_url = f"{base_url}/{chrom}.fasta.gz"
            local_file = self.data_dir / f"{chrom}.fasta.gz"

            if self.download_with_progress(fasta_url, local_file, f"Chromosome {chrom}"):
                success_count += 1
            else:
                print(f"Failed to download {chrom}, continuing...")

        if success_count > 0:
            print(f"Successfully downloaded {success_count}/{total_chromosomes} chromosomes")
            print("You can concatenate these files for a complete genome:")
            print(f"cat {self.data_dir}/*.fasta.gz > {self.data_dir}/human_complete.fasta.gz")
            return True

        return False

    def download_ecoli(self) -> bool:
        """Download E. coli K-12 genome."""
        print("=== Downloading E. coli K-12 MG1655 ===")

        # Direct download from NCBI
        accession = self.genomes['ecoli']['accession']
        url = f"https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/Escherichia_coli/reference/GCF_000005845.2_ASM584v2/{accession}_ASM584v2_genomic.fna.gz"
        local_file = self.data_dir / "ecoli.fasta.gz"

        if self.download_with_progress(url, local_file, "E. coli K-12 genome"):
            # Decompress
            print("Decompressing...")
            import gzip
            with gzip.open(local_file, 'rb') as f_in:
                with open(self.data_dir / "ecoli.fasta", 'wb') as f_out:
                    f_out.write(f_in.read())
            local_file.unlink()  # Remove compressed file
            print(f"E. coli genome saved as: {self.data_dir / 'ecoli.fasta'}")
            return True

        return False

    def download_yeast(self) -> bool:
        """Download yeast genome."""
        print("=== Downloading Saccharomyces cerevisiae S288C ===")

        accession = self.genomes['yeast']['accession']
        url = f"https://ftp.ncbi.nlm.nih.gov/genomes/refseq/fungi/Saccharomyces_cerevisiae/reference/GCF_000146045.2_R64/{accession}_R64_genomic.fna.gz"
        local_file = self.data_dir / "yeast.fasta.gz"

        if self.download_with_progress(url, local_file, "Yeast genome"):
            # Decompress
            print("Decompressing...")
            import gzip
            with gzip.open(local_file, 'rb') as f_in:
                with open(self.data_dir / "yeast.fasta", 'wb') as f_out:
                    f_out.write(f_in.read())
            local_file.unlink()
            print(f"Yeast genome saved as: {self.data_dir / 'yeast.fasta'}")
            return True

        return False

    def download_custom(self, url: str, filename: Optional[str] = None) -> bool:
        """Download a custom genome from URL."""
        print(f"=== Downloading Custom Genome ===")
        print(f"URL: {url}")

        if filename is None:
            filename = url.split('/')[-1]
            if not filename.endswith('.fasta'):
                filename += '.fasta'

        local_file = self.data_dir / filename

        return self.download_with_progress(url, local_file, f"Custom genome ({filename})")

    def list_available_genomes(self):
        """List all available genomes for download."""
        print("=== Available Genomes ===")
        for key, info in self.genomes.items():
            print(f"\n{key.upper()}:")
            print(f"  Name: {info['name']}")
            print(f"  Accession: {info['accession']}")
            print(f"  Description: {info['description']}")
            print(f"  Chromosomes: {len(info['chromosomes'])}")

    def verify_installation(self) -> bool:
        """Verify that required tools are installed."""
        print("=== Verifying Installation ===")

        issues = []

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 7):
            issues.append(f"Python {python_version.major}.{python_version.minor} detected. Python 3.7+ required.")

        # Check required packages
        required_packages = ['numpy', 'vispy']
        for package in required_packages:
            try:
                __import__(package)
                print(f"✓ {package} available")
            except ImportError:
                issues.append(f"Missing required package: {package}")

        # Check optional packages
        optional_packages = ['Bio', 'ctypes']
        for package in optional_packages:
            try:
                __import__(package)
                print(f"✓ {package} available (optional)")
            except ImportError:
                print(f"⚠ {package} not available (optional)")

        # Check NCBI datasets tool
        try:
            result = subprocess.run(['datasets', '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✓ NCBI datasets CLI available")
            else:
                print("⚠ NCBI datasets CLI not available (manual download will be used)")
        except:
            print("⚠ NCBI datasets CLI not available (manual download will be used)")

        if issues:
            print("\n❌ Installation Issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("\n✅ All requirements satisfied!")
            return True

def main():
    parser = argparse.ArgumentParser(
        description="Genome Data Loader for Human Genome Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python genome_loader.py --human-reference    # Download human genome
  python genome_loader.py --ecoli             # Download E. coli genome
  python genome_loader.py --yeast             # Download yeast genome
  python genome_loader.py --list              # List available genomes
  python genome_loader.py --verify            # Check installation
  python genome_loader.py --custom URL        # Download custom genome

Environment Variables:
  GENOME_LIMIT      Maximum nucleotides to load (default: 100000)
  GENOME_CHROMOSOME Specific chromosome (e.g., 'NC_000001.11')
  GENOME_START      Starting position in sequence (default: 0)
        """
    )

    parser.add_argument('--human-reference', action='store_true',
                       help='Download human genome reference (GRCh38)')
    parser.add_argument('--ecoli', action='store_true',
                       help='Download E. coli K-12 genome')
    parser.add_argument('--yeast', action='store_true',
                       help='Download yeast genome')
    parser.add_argument('--custom', metavar='URL',
                       help='Download custom genome from URL')
    parser.add_argument('--filename', help='Custom filename for download')
    parser.add_argument('--list', action='store_true',
                       help='List available genomes')
    parser.add_argument('--verify', action='store_true',
                       help='Verify installation requirements')
    parser.add_argument('--data-dir', default='data',
                       help='Data directory (default: data)')

    args = parser.parse_args()

    loader = GenomeLoader(args.data_dir)

    if args.verify:
        success = loader.verify_installation()
        sys.exit(0 if success else 1)

    if args.list:
        loader.list_available_genomes()
        return

    if args.human_reference:
        success = loader.download_human_reference()
    elif args.ecoli:
        success = loader.download_ecoli()
    elif args.yeast:
        success = loader.download_yeast()
    elif args.custom:
        success = loader.download_custom(args.custom, args.filename)
    else:
        parser.print_help()
        return

    if success:
        print("\n✅ Download completed successfully!")
        print(f"Data saved in: {loader.data_dir}")
        print("\nYou can now run visualization scripts with:")
        print("python human_genome_control_panel.py")
    else:
        print("\n❌ Download failed. Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()