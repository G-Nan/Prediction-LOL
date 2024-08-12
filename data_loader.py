import os
import gdown
import zipfile

def data_download(data_type):

    dic_download_url = {
        'RAW': 'https://drive.google.com/uc?id=18TILYfqaHot4d-JeU781xzoWH8wj6Z4w',
        'ALL': 'https://drive.google.com/uc?id=1vz08BMj8sQTS4pm5TV3cZhljlwWRwThP',
        'MEAN': 'https://drive.google.com/uc?id=1EclG4SoW-FUK_VQK4HQLheLIUnK9zMHY',
        'WEIGHTEDMEAN': 'https://drive.google.com/uc?id=1IwnO-XnwV8tReK_V8CnY0bLWKW_1-RJu',
        'POINT': 'https://drive.google.com/uc?id=1DMZA_PHkxK5gcQei3h9kVoz5Vj2yd8fA',
        'TIMESERIES': 'https://drive.google.com/uc?id=1RWONLW0DOjAZV7kXwew3_YTd1iLrnTbW',
        'LANCHESTER': 'https://drive.google.com/uc?id=1SvMj6N071FKZYq8WZfovbLZpTsXq810M'        
    }
   
    if not os.path.exists('Data/'):
        os.makedirs('Data/')
    
    for dt in data_type:
            
        url = dic_download_url[dt]
        output = f'Data/{dt}.zip'
        gdown.download(url, output, quiet = False)
            
        output_dir = output[:-4]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with zipfile.ZipFile(output, 'r') as zip_ref:
            print('Extracting...')
            zip_ref.extractall(output_dir)