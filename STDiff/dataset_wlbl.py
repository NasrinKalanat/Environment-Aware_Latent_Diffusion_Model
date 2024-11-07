# All folders
import os
import numpy as np
import pandas as pd 
import glob
import xml.etree.ElementTree as ET
import datetime
from datetime import datetime, timedelta, timezone
from skimage.metrics import mean_squared_error, structural_similarity, normalized_root_mse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MultiLabelBinarizer
import h5py
import requests
from PIL import Image
import torch
from  torchvision import transforms
import joblib
import json
# def download_image(url, file_path, file_name=""):
#   # print("downloading "+url)
#   full_path = file_path + file_name
#   res = requests.get(url, stream = True)
#   if res.status_code == 200:
#     with open(full_path,'wb') as f:
#         shutil.copyfileobj(res.raw, f)
#     # print('Image sucessfully Downloaded: ',file_name)
#   else:
#     print('Image Couldn\'t be retrieved '+ res.status_code)

def download_image(url, file_path, file_name=""):
    # download from image url and import it as a numpy array
    full_path = file_path + file_name
    res = requests.get(url, stream=True) # get full image
    if res.status_code == 200:
      img=res.raw
      img = Image.open(img)
      img = img.resize((360,360))
      # img = ImageOps.grayscale(img) # grayscale
      img = img.tobytes() # convert to bytes
      img = bytearray(img) # create byte array
      img = np.asarray(img, dtype="uint8") # 360x360 array
      img = img.reshape(360, 360, 3)
      np.save(full_path,img)

      # with open(full_path, 'rb') as f:
      #   img = np.load(full_path)
      #   img=img.astype('float32') / 255
      # fig,ax = plt.subplots(1)
      # ax.imshow(img)

    else:
      print('Image Couldn\'t be retrieved '+ res.status_code)
    return img

class ImageNetVidDataset(torch.utils.data.Dataset):

  def __init__(self, image_size=256, batch_size=2 , len_seq=8, path="", path_weather="", path_scaler="", normalize_flag=True, phase="train", transform=None, mask_frac=1):

    self.phase=phase
    self.batch_size=batch_size
    self.len_seq=len_seq
    self.l_seq=batch_size*len_seq
    self.image_size=image_size
    self.path_weather=path_weather
    self.transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.CenterCrop(340),
    # transforms.Resize(360, Image.BICUBIC),
    transforms.Resize((image_size,image_size)),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # torch.squeeze,
    # np.array
    ])
    predefined_classes=["Sunny/Clear", "Cloudy/Overcast", "Rainy", "Snowy", "Foggy/Misty", "Windy", "Stormy/Severe", "Hot/Heatwave", "Cold/Cold Wave", "Mixed/Variable"]
    self.num_classes=len(predefined_classes)

    self.images,self.labels,self.dates,self.weather,self.weather_label = [], [], [], [],[]
    self.boundries = []

    fol_list = os.listdir(path)
    fol_name = ""
    if phase == "test":
      fol_name = "Avery Brook_River Right_01171000"
      # fol_list = ["Avery Brook_Bridge_01171000"]
      # fol_list = ["West Brook Upper_01171030"]
      # fol_list = ["West Brook Lower_01171090"]
      # fol_list = ["Avery Brook_River Left_01171000"]
      # fol_list = ["Avery Brook_River Right_01171000"]
      # fol_list = ['Obear Brook Lower_01171070']
      # fol_list = ["West Brook Reservoir_01171020"]
      fol_name = "_"+"ar"
 
    weather_stat=[]
    for fol in fol_list:
        # load
        images, temps, dates, weather, weather_label, w_stat = self.load_data(path, fol)
        weather_stat.append(w_stat)
        print(fol,path)

        # sort
        images, temps, dates, weather, weather_label = self.sort_data(images, temps, dates, weather, weather_label)

        # # normalize
        # self.normalizer = StandardScaler()
        # temps = self.normalizer.fit_transform(temps)
        # joblib.dump(self.normalizer, os.path.join(path_scaler, "flow_scaler_"+phase))

        # self.wnormalizer = StandardScaler()
        # weather = self.wnormalizer.fit_transform(weather)
        # joblib.dump(self.wnormalizer, os.path.join(path_scaler, "weather_scaler_"+phase))


        # self.timetransformer=MinMaxScaler()
        # dates =self.timetransformer.fit_transform(dates)
        # joblib.dump(self.timetransformer, os.path.join(path_scaler, "time_scaler_"+phase))

        # self.timestamps=dates
        # # temps=np.log(temps+1)

        # # mask
        # random_indices = np.random.choice(temps.shape[0], size=int(temps.shape[0] * mask_frac), replace=False)
        # temps[random_indices,:]=[None]

        images, temps, dates, weather, weather_label=self.generate_many2many_data2(self.l_seq, images, temps, dates, weather, weather_label)
        if images.shape[0]>0:
          # split to train, val, test
          images,labels,dates,weather,weather_label=self.data_split(images,temps,dates,weather,weather_label,self.phase,mask_frac)
          # self.images,self.labels,self.dates=self.generate_many2many_data_split(self.l_seq, images, temps, dates, phase)
          start = len(self.images)
          self.images.append(images)
          end = len(self.images) - 1
          self.boundries.append((start, end))
          self.labels.append(labels)
          self.dates.append(dates)
          self.weather.append(weather)
          self.weather_label.append(weather_label)
          print(fol, phase)
          print(pd.to_datetime(np.min(dates), unit='s'), pd.to_datetime(np.max(dates), unit='s'))
          print(np.min(labels), np.max(labels))
          print(len(images))

    weather_stat = pd.concat(weather_stat, ignore_index=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
      print(weather_stat.describe())

    self.images=np.concatenate(self.images,axis=0)
    self.boundries.append((start, end))
    self.labels=np.concatenate(self.labels,axis=0)
    self.dates=np.concatenate(self.dates,axis=0)
    self.weather=np.concatenate(self.weather,axis=0)
    self.weather_label= np.concatenate(self.weather_label, axis=0)
    print(self.weather.shape)

    # normalize
    if normalize_flag:
        self.normalizer = StandardScaler()
        temps = self.normalizer.fit_transform(self.labels.reshape(self.labels.shape[0]*self.labels.shape[1],-1))
        self.labels=temps.reshape(self.labels.shape[0], self.labels.shape[1], -1)
        joblib.dump(self.normalizer, os.path.join(path_scaler, "flow_scaler_"+phase+fol_name))

        self.wnormalizer = StandardScaler()
        weather = self.wnormalizer.fit_transform(self.weather.reshape(self.weather.shape[0]*self.weather.shape[1],-1))
        self.weather=weather.reshape(self.weather.shape[0], self.weather.shape[1], -1)
        joblib.dump(self.wnormalizer, os.path.join(path_scaler, "weather_scaler_"+phase+fol_name))

        self.wBinarizer = MultiLabelBinarizer(classes=predefined_classes)
        weather_label = self.wBinarizer.fit_transform(self.weather_label.reshape(self.weather_label.shape[0]*self.weather_label.shape[1]))
        self.weather_label=weather_label.reshape(self.weather_label.shape[0], self.weather_label.shape[1], -1)
        joblib.dump(self.wBinarizer, os.path.join(path_scaler, "weatherlabel_scaler_"+phase+fol_name))
        # print(self.wBinarizer.classes_)

        self.timetransformer=MinMaxScaler()
        dates =self.timetransformer.fit_transform(self.dates.reshape(self.dates.shape[0]*self.dates.shape[1],-1))
        self.dates=dates.reshape(self.dates.shape[0], self.dates.shape[1], -1)
        joblib.dump(self.timetransformer, os.path.join(path_scaler, "time_scaler_"+phase+fol_name))

        self.timestamps=dates

    # # download images (do just for first time)
    # if not os.path.exists(cur_path+"/"+fol+"/images/"):
    #   os.mkdir(cur_path+"/"+fol+"/images/")
    # delay=0.001
    # for i, (id,url) in enumerate(zip(self.imgs.image_id,self.imgs.url)):
    #   count=0
    #   while True:
    #       try:
    #         download_image(url,cur_path+fol+"/images/"+ str(id)+".jpg")
    #         self.imgs.ix[i,"image_path"]=cur_path+fol+"/images/"+ str(id)+".jpg"
    # #         download_image2(url,cur_path+fol+"/images/"+ str(id)+".npy")
    # #         self.imgs.loc[:,"image_path"].iloc[i]=cur_path+fol+"/images/"+ str(id)+".npy"
    #         time.sleep(delay)
    #         break
    #       except Exception as e:
    #         # raise e
    #         print("error: ", e, "url: ",url)
    #         if count>5:
    #             break
    #         count+=1
    #         time.sleep(10*count)

  def half_up_minute(self, x):
    delta = timedelta(minutes=15)
    ref_time = datetime(1970,1,1, tzinfo=x.tzinfo)
    return ref_time + round((x - ref_time) / delta) * delta

  def _loadimage(self, path, url):
    # # load ".jpg" files
    # with open(path, 'rb') as f:
    #   img = Image.open(f)
    #   return img.convert('RGB')
    try:
      with open(path, 'rb') as f:
        img = np.load(path,allow_pickle=True)
        return img
    except:
      img = download_image(url,path)
      with open(path, 'rb') as f:
        img = np.load(path,allow_pickle=True)
        return img

  def denormalize(self, pred):
    return self.normalizer.inverse_transform(pred)

  def load_data_from_h5(h5Path):
      f = h5py.File(h5Path, 'r')
      images = f['image']
      temps = f['temps']
      depths = f['depths']
      dates = f['dates']

      return images, temps, depths, dates

  def load_data(self, path, fol):
      imgfile = pd.read_csv(path + fol + '/images.csv',dtype={'station_name':str,'station_id':int,'image_id':int,'timestamp':str,'filename':str,'url':str}, parse_dates=['timestamp'])
      valuesfile = pd.read_csv(path + fol + '/values.csv',dtype={'station_name':str,'station_id':int,'dataset_id':int,'series_id':int,'variable_id':str,'timestamp':str,'value':float}, parse_dates=['timestamp'])
      stationfile = pd.read_csv(path + fol + '/station.csv')

      all_files = glob.glob(os.path.join(self.path_weather, 'Weather', '*.xlsx'))
      weatherfile=[]
      for w in all_files:
        file = pd.read_excel(os.path.join(self.path_weather,"Weather",w), skiprows=[0,1,3], parse_dates=["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."])
#         file=pd.read_csv(os.path.join(root_data,"Weather",w), skiprows=[0,1,3], parse_dates=["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."])
#         file.columns=file.iloc[1]
#         file.drop([0,1,2], inplace=True)
#         print("nan", len(file[file.isna().any(axis=1)]))
        file.dropna(inplace=True)
        file.reset_index(drop=True, inplace=True)
#         file[["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]]=file[["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]].apply(pd.to_datetime)
        file[["Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]]=file[["Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]].applymap(datetime.timestamp)
        weatherfile.append(file)
      weatherfile=pd.concat(weatherfile, ignore_index=True)
#       weatherfile.rename(columns={"TIMESTAMP": "timestamp"}, inplace=True)

#       weatherfile[(weatherfile["Site_Name"]==fol.split("_")[0]) & (weatherfile["Station_No"]==fol.split("_")[1])]
#       weatherfile=weatherfile[["DateTime_EST", "GageHeight_Hobo_ft", "Discharge_Hobo_cfs", "WaterTemperature_HOBO_DegF"]]
#       weatherfile["DateTime_EST"]=pd.to_datetime(weatherfile["DateTime_EST"])

      # preprocessing time
      imgfile['timestamp'] = imgfile['timestamp'].map(self.half_up_minute)
      valuesfile['timestamp'] = valuesfile['timestamp'].map(self.half_up_minute)
#       weatherfile["DateTime_EST"]=weatherfile["DateTime_EST"].map(self.half_up_minute)
#       weatherfile[["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]]=weatherfile[["TIMESTAMP", "Time of Daily Temp Max", "Time of Min. Temp", "Time of Max Wind Spd", "Time of Min. Wind Spd."]].applymap(self.half_up_minute)

      if self.phase=="pretrain":
        data=imgfile.copy()
        data['value']=valuesfile['value'].mean()
      else:
        data = imgfile.merge(valuesfile,on=["station_id","timestamp"])

      def custom_filter(x):
            day_rows=x[(x["timestamp"].dt.hour>9) & (x["timestamp"].dt.hour<18)]
            if not day_rows.empty:
                return day_rows.head(1)
            return x.head(1)
#       data["date_tmp"]=data["timestamp"].dt.date
      data=data.groupby(data["timestamp"].dt.date, as_index=False).apply(custom_filter).reset_index(drop=True)

      data["date_tmp"]=data["timestamp"].dt.strftime('%Y-%m-%d')
      # print(data["date_tmp"].head(1))
      weatherfile["date_tmp"]=weatherfile["TIMESTAMP"].dt.strftime('%Y-%m-%d')
      # print(weatherfile["date_tmp"].head(1))
      weatherfile["TIMESTAMP"]=weatherfile["TIMESTAMP"].apply(datetime.timestamp)

      weatherfile=weatherfile.drop_duplicates(subset=['date_tmp']).reset_index(drop=True)
      # weatherlabelfile = pd.read_json(path + fol + '/response.jsonl', lines=True)
      l = []
      with open(self.path_weather+"response.jsonl", 'r') as weatherlabelfile:
          for line in weatherlabelfile.readlines():
              import ast
              t_jdata=json.loads(json.loads(line)[0]["messages"][1]["content"][51:])["Timestamp"]
              w_jdata=json.loads(json.loads(line)[1]["choices"][0]["message"]["content"])["Weather Classified Categories"]
              l.append([t_jdata, w_jdata])
      l_df=pd.DataFrame(data=l,columns=["date_tmp","weather_label"])
      l_df["date_tmp"]=pd.to_datetime(l_df["date_tmp"]).dt.strftime('%Y-%m-%d')
      # print(len(weatherfile),len(l_df))
      weatherfile = weatherfile.merge(l_df, on=["date_tmp"])
      # print(len(weatherfile), len(l_df))
      # weatherfile.drop("Timestamp",axis=1,inplace=True)
#       exp_d=pd.date_range(pd.to_datetime("2018-01-01"),pd.to_datetime("2023-05-14"))
#       print(list(set(exp_d)-set(weatherfile["date_tmp"])))
      # print("before", len(data))
      data = data.merge(weatherfile,on=["date_tmp"])
      # print("after", len(data))
      data.drop("date_tmp",axis=1,inplace=True)
      weatherfile.drop("date_tmp",axis=1,inplace=True)
#       print(data.iloc[10:20])

      min_time=min(imgfile['timestamp'])
      max_time=max(imgfile['timestamp'])
      self.num_days=(max_time-min_time).days
      times=data['timestamp'].apply(datetime.timestamp)
#       times=data['timestamp'].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

      images=[]
      temps=[]
      dates=[]
      tmp=[]
      # wlabel=[]
      for i, (id,url,v,t,lbl) in enumerate(zip(data.image_id,data.url,data.value,times,data.weather_label)):
        # images.append(np.array(Image.fromarray(self._loadimage(cur_path+fol+"/images/"+ str(id)+".npy",url)).convert('L')))
        # img_cur=self._loadimage(cur_path+fol+"/images/"+ str(id)+".npy",url)
        # if self.transform is not None:
        #   img_cur=self.transform(img_cur)

        images.append([path+fol+"/images/"+ str(id)+".npy",url,id])
        temps.append([v])
        dates.append([t])
        # wlabel.append(lbl)

      # # import h5py
      #   tmp.append(np.array(Image.fromarray(self._loadimage(cur_path+fol+"/images/"+ str(id)+".npy",url)).convert('L')))
      #   depths.append([0])

      # with h5py.File('/content/drive/My Drive/spatial_temporal/stream_img/fpe-westbrook/h5_2019.h5','w') as f:
      #   f['image'] = np.array(tmp)
      #   f['dates'] = np.array(dates)
      #   f['temps'] = np.array(temps)
      #   f['depths'] = np.array(depths)

      dates=np.array(dates)
      images=np.array(images)
      temps=np.array(temps)
      # wlabel=np.array(wlabel)
      # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
      #   print(data[weatherfile.columns].describe().transpose()[['min','max','mean','std']])
      return images, temps, dates, data[weatherfile.columns].drop(columns=["weather_label"],axis=1).values, data["weather_label"].values, weatherfile

  def sort_data(self, images, temps, dates, weather, weather_label):
      index = np.argsort(dates, axis=0)
      index = index.reshape(index.shape[0],)
      return images[index], temps[index], dates[index], weather[index], weather_label[index]

  def generate_many2many_data2(self, time_step, images, temps, dates, weather, weather_label):
      img_len = images.shape[0]
      train_x = []
      train_y = []
      train_d = []
      train_weather = []
      train_weather_label = []
      for i in range(0, img_len - time_step, time_step):   # non-overlap
          train_x.append(images[i : i + time_step+1][:])
          train_y.append(temps[i : i + time_step+1][:])
          # train_y.append(np.concatenate((temps[i : i + time_step][:], depths[i : i + time_step][:]), axis=1))
          train_d.append(dates[i : i + time_step+1][:])
          train_weather.append(weather[i : i + time_step+1][:])
          train_weather_label.append(weather_label[i : i + time_step+1][:])
      train_x = np.array(train_x)
      train_y = np.array(train_y)
      train_d = np.array(train_d)
      train_weather=np.array(train_weather)
      train_weather_label=np.array(train_weather_label)

      return train_x, train_y, train_d, train_weather, train_weather_label

  def generate_many2many_data_split(self, time_step, images, temps, dates, phase):
      img_len = images.shape[0]
      train_x = []
      train_y = []
      train_d = []
      if phase=="train":
        for i in range(0, img_len - time_step + 1, 2*time_step):   # non-overlap
            train_x.append(images[i : i + time_step][:])
            train_y.append(temps[i : i + time_step][:])
            train_d.append(dates[i : i + time_step][:])
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_d = np.array(train_d)
      elif phase=="val":
        for i in range(time_step, img_len - time_step + 1, 2*time_step):   # non-overlap
            train_x.append(images[i : i + int(time_step/2)][:])
            train_y.append(temps[i : i + int(time_step/2)][:])
            train_d.append(dates[i : i + int(time_step/2)][:])
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_d = np.array(train_d)
      else:
        for i in range(int(3*time_step/2), img_len - time_step + 1, 2*time_step):   # non-overlap
            train_x.append(images[i : i + int(time_step/2)][:])
            train_y.append(temps[i : i + int(time_step/2)][:])
            train_d.append(dates[i : i + int(time_step/2)][:])
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_d = np.array(train_d)
      return train_x, train_y, train_d

  def data_split(self, t_x,t_y,t_d,weather,weather_label,phase,mask_frac):
      np.random.seed(42)
      indices = np.arange(t_x.shape[0])
      np.random.shuffle(indices)
      np.random.seed(None)

      # Define split sizes
      train_size = int(0.8 * t_x.shape[0])
      val_size = int(0.2 * t_x.shape[0])
                       
      if phase=="trainval":
        if not mask_frac:
          train_x1 = t_x[:round(t_x.shape[0] * 0.375)]
          train_y1 = t_y[:round(t_y.shape[0] * 0.375)]
          train_d1 = t_d[:round(t_d.shape[0] * 0.375)]
          train_x2 = t_x[round(t_x.shape[0] * 0.675):]
          train_y2 = t_y[round(t_y.shape[0] * 0.675):]
          train_d2 = t_d[round(t_d.shape[0] * 0.675):]
          train_w1 = weather[:round(weather.shape[0] * 0.375)]
          train_w2 = weather[round(weather.shape[0] * 0.675):]
          train_lbl1 = weather_label[:round(weather_label.shape[0] * 0.375)]
          train_lbl2 = weather_label[round(weather_label.shape[0] * 0.675):]

          imgs = np.concatenate((train_x1, train_x2), axis=0)
          labels = np.concatenate((train_y1, train_y2), axis=0)
          dates = np.concatenate((train_d1, train_d2), axis=0)
          weather = np.concatenate((train_w1, train_w2), axis=0)
          weather_label = np.concatenate((train_lbl1, train_lbl2), axis=0)
        else:      
          train_indices = indices[:train_size]
          imgs = t_x[train_indices]
          labels = t_y[train_indices]
          dates = t_d[train_indices]
          weather = weather[train_indices]
          weather_label = weather_label[train_indices]

      elif phase=="train":
        if not mask_frac:
          train_x1 = t_x[:round(t_x.shape[0] * 0.375)]
          train_y1 = t_y[:round(t_y.shape[0] * 0.375)]
          train_d1 = t_d[:round(t_d.shape[0] * 0.375)]
          train_x2 = t_x[round(t_x.shape[0] * 0.875):]
          train_y2 = t_y[round(t_y.shape[0] * 0.875):]
          train_d2 = t_d[round(t_d.shape[0] * 0.875):]
          train_w1 = weather[:round(weather.shape[0] * 0.375)]
          train_w2 = weather[round(weather.shape[0] * 0.875):]
          train_lbl1 = weather_label[:round(weather_label.shape[0] * 0.375)]
          train_lbl2 = weather_label[round(weather_label.shape[0] * 0.875):]

          imgs = np.concatenate((train_x1, train_x2), axis=0)
          labels = np.concatenate((train_y1, train_y2), axis=0)
          dates = np.concatenate((train_d1, train_d2), axis=0)
          weather = np.concatenate((train_w1, train_w2), axis=0)
          weather_label = np.concatenate((train_lbl1, train_lbl2), axis=0)
        else:      
          train_indices = indices[:train_size]
          imgs = t_x[train_indices]
          labels = t_y[train_indices]
          dates = t_d[train_indices]
          weather = weather[train_indices]
          weather_label = weather_label[train_indices]

        # if mask_frac==-1:
        #     # mask sequentially
        #     labels[:round(t_y.shape[0] * 0.375),:,:] = [None]
        # else:
        #     # mask randomly
        #     random_indices = np.random.choice(labels.size, size=int(labels.size * mask_frac), replace=False)
        #     row_indices, col_indices = random_indices//labels.shape[1], random_indices%labels.shape[1]
        #     labels[row_indices, col_indices,:]=[None]

      elif phase=="val":
        if not mask_frac:
          imgs = t_x[round(t_x.shape[0] * 0.675):round(t_x.shape[0] * 0.875)]
          labels = t_y[round(t_y.shape[0] * 0.675):round(t_x.shape[0] * 0.875)]
          dates = t_d[round(t_d.shape[0] * 0.675):round(t_x.shape[0] * 0.875)]
          weather = weather[round(weather.shape[0] * 0.675):round(weather.shape[0] * 0.875)]
          weather_label = weather_label[round(weather_label.shape[0] * 0.675):round(weather_label.shape[0] * 0.875)]
        else:
          val_indices = indices[train_size:train_size + val_size]
          imgs = t_x[val_indices]
          labels = t_y[val_indices]
          dates = t_d[val_indices]
          weather = weather[val_indices]
          weather_label = weather_label[val_indices]

      elif phase=="test":
        if not mask_frac:
          imgs = t_x[round(t_x.shape[0] * 0.375):round(t_x.shape[0] * 0.675)]
          labels = t_y[round(t_y.shape[0] * 0.375):round(t_x.shape[0] * 0.675)]
          dates = t_d[round(t_d.shape[0] * 0.375):round(t_x.shape[0] * 0.675)]
          weather=weather[round(weather.shape[0] * 0.375):round(weather.shape[0] * 0.675)]
          weather_label=weather_label[round(weather_label.shape[0] * 0.375):round(weather_label.shape[0] * 0.675)]
        else:
          # test_indices = indices[train_size + val_size:]
          test_indices = indices[train_size:]
          imgs = t_x[test_indices]
          labels = t_y[test_indices]
          dates = t_d[test_indices]
          weather = weather[test_indices]
          weather_label = weather_label[test_indices]
      else:
          return t_x, t_y, t_d, weather, weather_label
      return imgs, labels, dates, weather, weather_label

  def __getitem__(self, id):
    # print(id)
    area_id = 0
    for i, (s,e) in enumerate(self.boundries):
        if s<=id<=e:
            area_id = i
    # id_random = np.random.randint(0, len(self)-(self.boundries[area_id][1]-self.boundries[area_id][0]+1))
    # if id_random>=self.boundries[area_id][0]:
    #     id_random += self.boundries[area_id][1]-self.boundries[area_id][0]+1
    random_range = np.concatenate((np.arange(0, self.boundries[area_id][0]), np.arange(self.boundries[area_id][1]+1, len(self))))
    id_random = np.random.choice(random_range)
    imgs=torch.zeros((self.l_seq+1,3,self.image_size,self.image_size),dtype=torch.float32)
    imgs_random = torch.zeros((self.l_seq + 1, 3, self.image_size, self.image_size), dtype=torch.float32)
    # imgs=torch.zeros((self.l_seq,3,360,360),dtype=torch.float32)
    t=torch.zeros((self.l_seq+1,1),dtype=torch.float32)
    lbl=torch.zeros((self.l_seq+1,1),dtype=torch.float32)
    w = np.zeros((self.l_seq + 1, 16), dtype=np.float32)
    wlbl = np.zeros((self.l_seq + 1, self.num_classes), dtype=np.float32)
    ids=np.zeros((self.l_seq+1,1))
    for i in range(self.l_seq+1):
      cur_path,url=self.images[id][i][0],self.images[id][i][1]
      img_cur=self._loadimage(cur_path,url)
      if self.transform is not None:
        img_cur=self.transform(img_cur) # The pixel values of the tensor are of type float32 and range from 0 to 1 (transforms.ToTensor() method includes normalization as part of its functionality, by dividing the pixel values by 255 to rescale them to the range [0, 1]).
      imgs[i,...]=img_cur

      cur_path, url = self.images[id_random][i][0], self.images[id_random][i][1]
      img_cur = self._loadimage(cur_path, url)
      if self.transform is not None:
          img_cur = self.transform(
              img_cur)  # The pixel values of the tensor are of type float32 and range from 0 to 1 (transforms.ToTensor() method includes normalization as part of its functionality, by dividing the pixel values by 255 to rescale them to the range [0, 1]).
      imgs_random[i, ...] = img_cur

      t[i,...]=torch.tensor([self.dates[id][i][0]])
      lbl[i,...]=torch.tensor([self.labels[id][i][0]])
      w[i, ...] = np.array(self.weather[id][i][:])
      wlbl[i, ...] = np.array(self.weather_label[id][i][:])
      ids[i,...]=self.images[id][i][2]

#     return imgs.squeeze(), t, torch.from_numpy(self.labels[id].astype(np.float32)), w, torch.from_numpy(ids)
    w = torch.tensor(w)
    wlbl = torch.tensor(wlbl)
    images_nxt = imgs[1:]
    images_random = imgs_random[1:]
    t_nxt = t[1:]
    label_nxt = lbl[1:]
    weather_nxt = w[1:]
    wlabel_nxt = wlbl[1:]
    ids_nxt = ids[1:]

    images_X = imgs[:-1]
    t_X = t[:-1]
    label_X = lbl[:-1]
    weather_X = w[:-1]
    wlabel_X = wlbl[:-1]
    ids_X = ids[:-1]
    if self.l_seq == 1:
        return {"img": images_nxt.squeeze(0),
                "mixed": (images_X.squeeze(0), label_X.reshape(self.batch_size, self.len_seq, -1).squeeze(0),
                          weather_X.reshape(self.batch_size, self.len_seq, -1).squeeze(0), t_nxt.squeeze(0), label_nxt.squeeze(0), wlabel_nxt.squeeze(0), ids_nxt, images_random.squeeze(0))}
    return {"img": images_nxt, "mixed": (images_X, label_X.reshape(self.batch_size,self.len_seq,-1), weather_X.reshape(self.batch_size,self.len_seq,-1), t_nxt, label_nxt, wlabel_nxt, ids_nxt, images_random)}
    #     return {"img": images_nxt.squeeze(0),
    #             "mixed": {"cond": (images_X.squeeze(0), label_X.reshape(self.batch_size, self.len_seq, -1).squeeze(0), weather_X.reshape(self.batch_size, self.len_seq, -1).squeeze(0), 
    #                                t_nxt.squeeze(0), label_nxt.squeeze(0), wlabel_nxt.squeeze(0), ids_nxt),
    #                       "uncond": (images_random.squeeze(0), label_X.reshape(self.batch_size, self.len_seq, -1).squeeze(0), weather_X.reshape(self.batch_size, self.len_seq, -1).squeeze(0), 
    #                                  t_nxt.squeeze(0), label_nxt.squeeze(0), wlabel_nxt.squeeze(0), ids_nxt)}}
    # return {"img": images_nxt, "mixed": {"cond": (images_X, label_X.reshape(self.batch_size,self.len_seq,-1), weather_X.reshape(self.batch_size,self.len_seq,-1), t_nxt, label_nxt, wlabel_nxt, ids_nxt), 
    #                                      "uncond": (images_random, label_X.reshape(self.batch_size,self.len_seq,-1), weather_X.reshape(self.batch_size,self.len_seq,-1), t_nxt, label_nxt, wlabel_nxt, ids_nxt)}}

  def __len__(self):
      return self.images.shape[0]

