import gdown

# Google Drive file IDs (extracted from the URLs)
file_links = {
    "bills.csv": "1-iGX-Hd7yDP6NRgCfk-h-ywmVrbD-tsL",
    "venues.csv": "1-gysmauktGAlksZNc-XsowI3aUaYlO_C",
}

# Download each file
for filename, file_id in file_links.items():
    print(f"Downloading {filename}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", f"data/{filename}", quiet=False)

print("Download complete!")
