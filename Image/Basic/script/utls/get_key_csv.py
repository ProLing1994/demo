import pandas as pd
import json

if __name__ == "__main__":
    csv_path = "/mnt/huanyuan/temp/0517.csv"
    out_csv_path = "/mnt/huanyuan/temp/0517_out.csv"

    data_pd = pd.read_csv(csv_path, encoding='utf_8_sig')
    
    csv_list = []

    for idx, row in data_pd.iterrows():
        
        csv_dict = {}
        csv_dict['plate'] = json.loads(row['key'])['CEXINFO']['CEI']['NPINFO'][0]['VNUMBER']
        print(json.loads(row['key'])['CEXINFO']['CEI']['NPINFO'][0]['VNUMBER'])
        csv_list.append(csv_dict)

    csv_data_pd = pd.DataFrame(csv_list)
    csv_data_pd.to_csv(out_csv_path, index=False, encoding="utf_8_sig")