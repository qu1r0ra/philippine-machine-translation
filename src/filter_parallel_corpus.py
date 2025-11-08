import pandas as pd
import re

# Define file names
ceb_file_name = "translatewiki.ceb-es.ceb"
es_file_name = "translatewiki.ceb-es.es"

# Define output file names
out_ceb_file = "meaningful.ceb"
out_es_file = "meaningful.es"
out_csv_file = "meaningful_pairs.csv"

# Define filtering rules
min_words = 4

# Regexes to filter out UI text, placeholders, and markup
bad_patterns = [
    re.compile(r"\$\d"),  # Matches $1, $2, etc.
    re.compile(r"''+"),  # Matches '' (italics) or ''' (bold)
    re.compile(r"<[a-zA-Z/].*?>"),  # Matches HTML tags (<strong>, <i>, <nowiki>)
    re.compile(r"\[\[|\]\]"),  # Matches wiki link markup [[ or ]]
    re.compile(r"\[\w+://"),  # Matches external links [http://...]
    re.compile(r"\[\$\d"),  # Matches placeholders in links [$1
    re.compile(r"%\(.*\)."),  # Matches %(...)s placeholders
    re.compile(r"https?://"),  # Matches raw http/https URLs
    re.compile(r"^\(.*\)$"),  # Matches lines that are just parentheticals
    re.compile(r"Problema sa (ekspresyon|Lua)"),  # Matches specific error messages
    re.compile(r"scan failed"),  # Matches specific error messages
    re.compile(r"\{\{.*\}\}"),  # Matches {{PLURAL...}}
    re.compile(r"user_n"),  # Matches 'user_n'
    re.compile(r"^[A-Z][a-z]+$"),  # Matches single capitalized words (days/months)
    re.compile(r"^[a-z]+$"),  # Matches single lowercase words
    re.compile(r"^[A-Z]{2,3}$"),  # Matches 2-3 letter acronyms (DOM, LUN)
]

# Lists to store good pairs
good_ceb_sentences = []
good_es_sentences = []

try:
    # Read both files
    with open(ceb_file_name, "r", encoding="utf-8") as f_ceb:
        ceb_lines = f_ceb.readlines()

    with open(es_file_name, "r", encoding="utf-8") as f_es:
        es_lines = f_es.readlines()

    # Check if files are aligned
    if len(ceb_lines) != len(es_lines):
        print(
            f"Error: File line counts do not match. Cebuano: {len(ceb_lines)}, Spanish: {len(es_lines)}"
        )
    else:
        print(f"Processing {len(ceb_lines)} line pairs with strict filters...")

        # Iterate through pairs
        for i in range(len(ceb_lines)):
            ceb_line = ceb_lines[i].strip()
            es_line = es_lines[i].strip()

            # Check word count
            ceb_word_count = len(ceb_line.split())
            es_word_count = len(es_line.split())

            is_good = True

            # Apply filters
            if ceb_word_count < min_words or es_word_count < min_words:
                is_good = False
            else:
                # Check for bad patterns in both lines
                for pattern in bad_patterns:
                    if pattern.search(ceb_line) or pattern.search(es_line):
                        is_good = False
                        break  # No need to check other patterns for this line

            # If all checks passed, add to our lists
            if is_good:
                good_ceb_sentences.append(ceb_line)
                good_es_sentences.append(es_line)

        print(
            f"Found {len(good_ceb_sentences)} meaningful sentence pairs after strict filtering."
        )

        # Write to new .ceb and .es files
        with open(out_ceb_file, "w", encoding="utf-8") as f_out_ceb:
            for line in good_ceb_sentences:
                f_out_ceb.write(line + "\n")

        with open(out_es_file, "w", encoding="utf-8") as f_out_es:
            for line in good_es_sentences:
                f_out_es.write(line + "\n")

        print(f"Successfully wrote to {out_ceb_file} and {out_es_file}.")

        # Create and save DataFrame to CSV
        df_pairs = pd.DataFrame(
            {"cebuano": good_ceb_sentences, "spanish": good_es_sentences}
        )

        df_pairs.to_csv(out_csv_file, index=False, encoding="utf-8")
        print(f"Successfully wrote to {out_csv_file}.")

except FileNotFoundError as e:
    print(f"Error: Could not find file - {e.filename}")
except Exception as e:
    print(f"An error occurred: {e}")
