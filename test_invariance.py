from preprocessing import json_loader
from postprocessing import json_creator
import os

data_path = './data'
rasputin = 'Rasputin/Hard.json'
df = json_loader.json_to_blockmasks(
    os.path.join(data_path, rasputin)
)
json_creator.blockmasks_to_json(df, "Expert.json")
