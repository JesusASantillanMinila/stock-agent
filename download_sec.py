import os
import requests
import json

def download_sec_tickers():
    url = "https://www.sec.gov/files/company_tickers.json"
    
    # The SEC requires a custom User-Agent in the header for API requests
    headers = {
        "User-Agent": "Jesus Santillan Minila alosantillan@gmail.com"
    }
    
    print("Downloading SEC ticker list...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    
    # Format the data as "TICKER - Company Name"
    ticker_list = []
    for item in data.values():
        ticker = item.get("ticker")
        title = item.get("title")
        if ticker and title:
            ticker_list.append(f"{ticker} - {title}")
            
    # Sort the list alphabetically
    ticker_list.sort()
    
    # Ensure the target directory exists
    output_dir = "sec_list"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the file path and save (mode 'w' will overwrite the file completely)
    file_path = os.path.join(output_dir, "SEC_List.json")
    with open(file_path, "w") as f:
        json.dump(ticker_list, f, indent=4)
        
    print(f"Successfully saved {len(ticker_list)} tickers to {file_path}!")

if __name__ == "__main__":
    download_sec_tickers()