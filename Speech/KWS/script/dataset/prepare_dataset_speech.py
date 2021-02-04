def get_hash_name(file_name):
    hash_name = file_name.strip().split('.')[0].split('_')[0]
    return hash_name
