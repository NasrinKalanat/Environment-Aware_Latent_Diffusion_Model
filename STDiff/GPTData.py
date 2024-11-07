from dataset import ImageNetVidDataset
import json
from datetime import datetime
import numpy as np
import torch
import glob
import os
import pandas as pd

def create_req_file(data, output):

    # data = {"title": "", "instructions": "", "", ""}
    MODEL_TYPE = "gpt-3.5-turbo-1106"
    SYSTEM_DESC = """\
    You are an advanced weather classification assistant that classifies weather conditions based on input sensor data. The input includes Timestamp (in UTC), Average Temp (in °C), Max Daily Temp (in °C), Min Temp (in °C), Wind Speed (in m/s), Wind Direction (in degrees), Max Wind Spd (in m/s), Minimum Wind Spd (in m/s), Mean Relative Humidity (in %), Atmospheric Pressure (in millibars), Mean Solar Radiation (in Watts per square meter), and Total Rainfall (in mm). Your task is to accurately classify the weather into appropriate categories, providing a detailed reasoning for each classified category along with the exact threshold values used. Output in the following JSON Format.
    CATEGORIES:
    {
      "Sunny/Clear": "Minimal cloud cover, high solar radiation",
      "Cloudy/Overcast": "Significant cloud cover, reduced solar radiation",
      "Rainy": "Precipitation in the form of rain, higher humidity",
      "Snowy": "Snowfall, often with lower temperatures",
      "Foggy/Misty": "Low visibility due to fog or mist, high humidity",
      "Windy": "High wind speeds, varying temperature/precipitation",
      "Stormy/Severe": "Severe weather like thunderstorms, hail",
      "Hot/Heatwave": "Extremely high temperatures, high solar radiation",
      "Cold/Cold Wave": "Extremely low temperatures",
      "Mixed/Variable": "Variable conditions within the same period"
    }

    FORMAT:
    {
        "Timestamp": "<Timestamp of Data>",
        "Weather Classified Categories": ["Category1", "Category2", ...],
        "Reasons": [
            "Category1: Your reasoning for Category1 including the exact threshold values.",
            "Category2: Your reasoning for Category2 including the exact threshold values.",
            ...
        ]
    }

    EXAMPLE:
    {
        "Timestamp": "2020-01-01T23:59:00Z",
        "Weather Classified Categories": ["Windy", "Rainy"],
        "Reasons": ["Windy: Wind Speed is above the threshold value of [X m/s], indicating windy conditions.", "Rainy: The Total Rainfall is above [Y mm], classifying the weather as rainy."]
    }

    """

    with open(output, 'w') as file:
        flag = True
        for i in range(len(data)):
            w={}
            w["Timestamp"]= np.datetime_as_string(np.datetime64(int(data.loc[i,"TIMESTAMP"]), "s"), timezone='UTC')
            w["Average Temp"]= float(data.loc[i,"Average Temp"])
            w["Max Daily Temp"]= float(data.loc[i,"Max Daily Temp"])
            w["Min Temp"]= float(data.loc[i,"Min Temp"])
            w["Wind Speed"]= float(data.loc[i,"Wind Speed"])
            w["Wind Direction"]= float(data.loc[i,"Wind Direction"])
            w["Max Wind Spd"]= float(data.loc[i,"Max Wind Spd"])
            w["Minimum Wind Spd"]= float(data.loc[i,"Minimum Wind Spd"])
            w["Mean Relative Humidity"]= float(data.loc[i,"Mean Relative Humidity"])
            w["Atmospheric Pressure"]= float(data.loc[i,"Atmospheric Pressure"])
            w["Mean Solar Radiation"]= float(data.loc[i,"Mean Solar Radiation"])
            w["Total Rainfall"]= float(data.loc[i,"Total Rainfall"])
        # for id, time, flow, weather in [("7", datetime.now().timestamp(), "1.2", ["1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3"]),("3", datetime.now().timestamp(), "0.4", ["1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3","1.3"])]:
        #     w = {"River Flow Value": float(flow), "Timestamp": np.datetime_as_string(np.datetime64(int(time), "s"), timezone='UTC'), "Average Temp": float(weather[1]),
        #          "Max Daily Temp": float(weather[2]), "Min Temp": float(weather[4]),
        #          "Wind Speed": float(weather[6]), "Wind Direction": float(weather[7]),
        #          "Max Wind Spd": float(weather[8]),
        #          "Minimum Wind Spd": float(weather[10]), "Mean Relative Humidity": float(weather[12]),
        #          "Atmospheric Pressure": float(weather[13]), "Mean Solar Radiation": float(weather[14]),
        #          "Total Rainfall": float(weather[15])}

            req = {"model": MODEL_TYPE,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": SYSTEM_DESC},
                                 {"role": "user",
                                  "content": "Classify the input sensor into a weather categories" + json.dumps(w),
                                  }]}
            # if flag:
            #     data["metadata"] = {"row_id": 1}
            #     flag = False

            string = json.dumps(req)
            file.write(string + "\n")

# class GPTDataset(ImageNetVidDataset):
#     def __init__(self, image_size=256,path="", path_weather="", path_scaler="", phase="all"):
#         super(GPTDataset, self).__init__(image_size=image_size, batch_size=1, len_seq=1, path=path, path_weather=path_weather, path_scaler=path_scaler, normalize_flag=False, phase=phase)
#
#     def __getitem__(self, id):
#         t=self.dates[id][0][0]
#         lbl=self.labels[id][0][0]
#         w = self.weather[id][0][:]
#         ids=self.images[id][0][2]
#
#         return ids, t, lbl, w

if __name__ == "__main__":
    # gpt_input=GPTDataset(path='/data/nak168/spatial_temporal/stream_img/data/fpe-westbrook/', path_weather='/data/nak168/spatial_temporal/stream_img/data/', phase="pretrain")
    # loader = torch.utils.data.DataLoader(gpt_input, shuffle=False, num_workers=0, batch_size=1)
    # loader = []
    path_weather = '/data/nak168/spatial_temporal/stream_img/data/'
    all_files = glob.glob(os.path.join(path_weather, 'Weather', '*.xlsx'))
    weatherfile = []
    for w in all_files:
        file = pd.read_excel(os.path.join(path_weather, "Weather", w), skiprows=[0, 1, 3],
                             parse_dates=["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp",
                                          "Time of Max Wind Spd", "Time of Min. Wind Spd."])
        #         file=pd.read_csv(os.path.join(root_data,"Weather",w), skiprows=[0,1,3], parse_dates=["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."])
        #         file.columns=file.iloc[1]
        #         file.drop([0,1,2], inplace=True)
        #         print("nan", len(file[file.isna().any(axis=1)]))
        file.dropna(inplace=True)
        file.reset_index(drop=True, inplace=True)
        #         file[["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]]=file[["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]].apply(pd.to_datetime)
        file[["Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]] = file[
            ["Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]].applymap(
            datetime.timestamp)
        weatherfile.append(file)
    weatherfile = pd.concat(weatherfile, ignore_index=True)

    weatherfile["date_tmp"] = weatherfile["TIMESTAMP"].dt.strftime('%Y-%m-%d')
    # print(weatherfile["date_tmp"].head(1))
    weatherfile["TIMESTAMP"] = weatherfile["TIMESTAMP"].apply(datetime.timestamp)

    weatherfile = weatherfile.drop_duplicates(subset=['date_tmp']).reset_index(drop=True)

    create_req_file(weatherfile, output='gpt_requests.jsonl')

# python examples/api_request_parallel_processor.py \
#   --requests_filepath ../latent-diffusion/gpt_requests.jsonl \
#   --save_filepath ../latent-diffusion/response.jsonl \
#   --request_url https://api.openai.com/v1/chat/completions \
#   --max_requests_per_minute 100 \
#   --max_tokens_per_minute 40000 \
#   --token_encoding_name p50k_base \
#
#   --max_attempts 2 \
#   --logging_level 20
