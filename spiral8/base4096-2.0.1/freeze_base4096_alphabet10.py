#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unicodedata
import json

# -------- CONFIG --------
OUTPUT_TEXT_FILE = "frozen_base4096_alphabet.txt"
OUTPUT_PY_FILE = "frozen_base4096_alphabet.py"
SEED = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "!@#$%^&*()-_+=[{]};:',\"<>?/`|~"
)
MAX_CHARS = 4096
AMBIGUOUS_SET = {"I", "l", "1", "O", "0"}  # optional stricter filtering

# -------- CLASSIFICATION --------
categories = {
    "whitespace": [],
    "control": [],
    "combining": [],
    "quotes_delimiters": [],
    "slashes_pipes": [],
    "angle_brackets": [],
    "bidi": [],
    "ambiguous": [],
    "safe": [],
}

quote_chars = set("'\"`“”‘’«»‹›‚„‟")
slash_chars = set("\\/|")
angle_chars = set("<>")

def classify(ch):
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = "UNKNOWN"
    code = f"U+{ord(ch):04X}"
    entry = (ch, code, name)

    if ch.isspace():
        categories["whitespace"].append(entry)
    elif unicodedata.category(ch)[0] == "C":  # control, format, etc.
        categories["control"].append(entry)
    elif unicodedata.combining(ch) != 0:
        categories["combining"].append(entry)
    elif ch in quote_chars:
        categories["quotes_delimiters"].append(entry)
    elif ch in slash_chars:
        categories["slashes_pipes"].append(entry)
    elif ch in angle_chars:
        categories["angle_brackets"].append(entry)
    elif unicodedata.bidirectional(ch) in ("R", "AL", "AN"):
        categories["bidi"].append(entry)
    elif ch in AMBIGUOUS_SET:
        categories["ambiguous"].append(entry)
    else:
        categories["safe"].append(entry)

# -------- CHARACTER FILTER --------
def is_valid_char(c):
    try:
        cp = ord(c)
        if 0xD800 <= cp <= 0xDFFF:  # skip surrogates
            return False

        name = unicodedata.name(c)
        if any(bad in name for bad in ['CONTROL','PRIVATE USE','UNASSIGNED','TAG']):
            return False

        cat = unicodedata.category(c)
        if cat in ['Zs','Cc','Cf','Cn','Co']:  # space, control, format, unassigned, private
            return False
        if cat in ['Mn','Mc','Me']:  # combining marks
            return False

        return True
    except ValueError:
        return False

# -------- ALPHABET GENERATION --------
def generate_frozen_base4096(seed):
    seen = set()
    safe_chars = []

    # Step 1: Add seed characters if they are safe
    for ch in seed:
        if ch in seen:
            continue
        if not is_valid_char(ch):
            continue
        classify(ch)
        if categories["safe"] and categories["safe"][-1][0] == ch:
            safe_chars.append(ch)
            seen.add(ch)

    # Step 2: Fill remaining slots from Unicode
    for codepoint in range(0x20, 0x110000):

        if 0xD800 <= codepoint <= 0xDFFF:  # skip surrogate area
            continue
        c = chr(codepoint)
        
        if len(safe_chars) >= MAX_CHARS:
            break
        c = chr(codepoint)
        if c in seen:
            continue
        if not is_valid_char(c):
            continue
        classify(c)
        if categories["safe"] and categories["safe"][-1][0] == c:
            safe_chars.append(c)
            seen.add(c)

    if len(safe_chars) != MAX_CHARS:
        raise ValueError(f"Only generated {len(safe_chars)} safe characters. Consider relaxing ambiguous filters.")

    return ''.join(safe_chars)

# -------- MAIN --------
def main():
    frozen_alphabet = generate_frozen_base4096(SEED)

    # Save plain text
    with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(frozen_alphabet)

    # Save Python constant file
    with open(OUTPUT_PY_FILE, "w", encoding="utf-8") as f:
        f.write("# frozen_base4096_alphabet.py\n")
        f.write("# Canonical Base-4096 Alphabet (fully visible, frozen, deterministic)\n\n")
        f.write("FROZEN_BASE4096_ALPHABET = (\n")
        for i in range(0, MAX_CHARS, 64):
            chunk = frozen_alphabet[i:i+64]
            f.write(f"    {json.dumps(chunk, ensure_ascii=False)}\n")
        f.write(")\n")

    # Print report
    print("✅ Fully visible canonical Base-4096 alphabet exported.")
    print(f"Length: {len(frozen_alphabet)}, Unique: {len(set(frozen_alphabet))}")

    print("\n=== Problematic Categories ===")
    for cat, items in categories.items():
        if cat == "safe" or not items:
            continue
        print(f"\n[{cat.upper()}] ({len(items)} found)")
        for ch, code, name in items:
            disp = ch if not ch.isspace() else repr(ch)
            print(f"  {disp}  {code}  {name}")

    print("\n=== SAFE SUBSET ===")
    safe_chars_str = "".join(ch for ch, _, _ in categories["safe"])
    print(safe_chars_str)
    print(f"\nTotal safe characters: {len(safe_chars_str)} / {MAX_CHARS}")

if __name__ == "__main__":
    main()
