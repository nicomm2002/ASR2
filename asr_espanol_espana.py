def download_common_voice_spain():
    # Adjusted download function for Common Voice Spain
    # Removed trust_remote_code=True parameter
    dataset = load_dataset('common_voice', 'es', split='train')
    return dataset


def download_rtve2022():
    # Adjusted download function for RTVE 2022
    # Removed trust_remote_code=True parameter
    dataset = load_dataset('rtve2022', split='train')
    return dataset


def load_rtve2022_hf():
    # Adjusted load function for RTVE 2022
    # Removed trust_remote_code=True parameter
    dataset = load_dataset('rtve2022', split='train')
    return dataset


def build_dataloaders():
    # Dataset loading and dataloader creation
    common_voice_data = download_common_voice_spain()
    rtve_data = download_rtve2022()
    
    if not common_voice_data and not rtve_data:
        raise ValueError("Real data is required. Please download either the Common Voice 16.0 ES dataset (filtered to peninsular speakers) or the RTVE 2022 ASR dataset, or both for better performance.")
    
    # Provide dataset sizes and statistics
    print(f"Common Voice dataset size: {len(common_voice_data)}")
    print(f"RTVE dataset size: {len(rtve_data)}")
    
    return common_voice_data, rtve_data
