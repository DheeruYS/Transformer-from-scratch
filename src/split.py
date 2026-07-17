from sklearn.model_selection import train_test_split
import csv

# Read and filter sentence pairs with < 100 words
max_length = 100
fi_filtered = []
en_filtered = []
with open("EUbookshop/EUbookshop.fi", encoding="utf-8") as f1, \
     open("EUbookshop/EUbookshop.en", encoding="utf-8") as f2:
    for fi_line, en_line in zip(f1, f2):
        fi_tokens = fi_line.strip().split()
        en_tokens = en_line.strip().split()
        if len(fi_tokens) < max_length and len(en_tokens) < max_length:
            fi_filtered.append(fi_line)
            en_filtered.append(en_line)

# Split into train/valid/test
fi_train, fi_temp, en_train, en_temp = train_test_split(fi_filtered, en_filtered, test_size=0.2, random_state=42)
fi_valid, fi_test, en_valid, en_test = train_test_split(fi_temp, en_temp, test_size=0.5, random_state=42)

def save(prefix, fi, en):
    with open(prefix+".fi","w",encoding="utf-8") as f: f.writelines(fi)
    with open(prefix+".en","w",encoding="utf-8") as f: f.writelines(en)

save("EUbookshop/train", fi_train, en_train)
save("EUbookshop/valid", fi_valid, en_valid)
save("EUbookshop/test", fi_test, en_test)

# Create TSV files for train/valid/test
for split in ["train", "valid", "test"]:
    fi_file = f"EUbookshop/{split}.fi"
    en_file = f"EUbookshop/{split}.en"
    out_file = f"EUbookshop/{split}.tsv"

    with open(fi_file, encoding="utf-8") as f1, \
         open(en_file, encoding="utf-8") as f2, \
         open(out_file, "w", encoding="utf-8", newline='') as out:
        writer = csv.writer(out, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["fi", "en"])  # header
        for fi_line, en_line in zip(f1, f2):
            fi_line = fi_line.strip().replace("\t", " ").replace("\n", " ")
            en_line = en_line.strip().replace("\t", " ").replace("\n", " ")
            if fi_line and en_line:
                writer.writerow([fi_line, en_line])