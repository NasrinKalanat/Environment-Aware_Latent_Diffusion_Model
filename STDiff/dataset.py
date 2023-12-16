# All folders
import os
import numpy as np
import pandas as pd 
import glob
import xml.etree.ElementTree as ET
import datetime
from datetime import datetime, timedelta, timezone
from skimage.metrics import mean_squared_error, structural_similarity, normalized_root_mse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import h5py
import requests
from PIL import Image
import torch
from  torchvision import transforms
import joblib
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

  def __init__(self, image_size=256, batch_size=2 , len_seq=8, path="", path_weather="", path_scaler="", phase="train", transform=None, mask_frac=0):

    self.phase=phase
    self.batch_size=batch_size
    self.l_seq=len_seq
    self.len_seq=batch_size*len_seq
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

    self.images,self.labels,self.dates,self.weather = [], [], [], []

    for fol in os.listdir(path):
        # load
        print(fol,path)
        images, temps, dates, weather=self.load_data(path, fol)

        # sort
        images, temps, dates, weather=self.sort_data(images, temps, dates, weather)

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

        images, temps, dates, weather=self.generate_many2many_data2(self.len_seq, images, temps, dates, weather)
        if images.shape[0]>0:
          # split to train, val, test
          images,labels,dates,weather=self.data_split(images,temps,dates,weather,self.phase,mask_frac)
          # self.images,self.labels,self.dates=self.generate_many2many_data_split(self.len_seq, images, temps, dates, phase)
          self.images.append(images)
          self.labels.append(labels)
          self.dates.append(dates)
          self.weather.append(weather)

    self.images=np.concatenate(self.images,axis=0)
    self.labels=np.concatenate(self.labels,axis=0)
    self.dates=np.concatenate(self.dates,axis=0)
    self.weather=np.concatenate(self.weather,axis=0)
    print(self.weather.shape)

    # normalize
    self.normalizer = StandardScaler()
    temps = self.normalizer.fit_transform(self.labels.reshape(self.labels.shape[0]*self.labels.shape[1],-1))
    self.labels=temps.reshape(self.labels.shape[0], self.labels.shape[1], -1)
    joblib.dump(self.normalizer, os.path.join(path_scaler, "flow_scaler_"+phase))

    self.wnormalizer = StandardScaler()
    weather = self.wnormalizer.fit_transform(self.weather.reshape(self.weather.shape[0]*self.weather.shape[1],-1))
    self.weather=weather.reshape(self.weather.shape[0], self.weather.shape[1], -1)
    joblib.dump(self.wnormalizer, os.path.join(path_scaler, "weather_scaler_"+phase))


    self.timetransformer=MinMaxScaler()
    dates =self.timetransformer.fit_transform(self.dates.reshape(self.dates.shape[0]*self.dates.shape[1],-1))
    self.dates=dates.reshape(self.dates.shape[0], self.dates.shape[1], -1)
    joblib.dump(self.timetransformer, os.path.join(path_scaler, "time_scaler_"+phase))

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

      if self.phase=="pre_train":
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
      depths=[]
      for i, (id,url,v,t) in enumerate(zip(data.image_id,data.url,data.value,times)):
        # images.append(np.array(Image.fromarray(self._loadimage(cur_path+fol+"/images/"+ str(id)+".npy",url)).convert('L')))
        # img_cur=self._loadimage(cur_path+fol+"/images/"+ str(id)+".npy",url)
        # if self.transform is not None:
        #   img_cur=self.transform(img_cur)

        images.append([path+fol+"/images/"+ str(id)+".npy",url,id])
        temps.append([v])
        dates.append([t])

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
      return images, temps, dates, data[weatherfile.columns].values


  def sort_data(self, images, temps, dates, weather):
      index = np.argsort(dates, axis=0)
      index = index.reshape(index.shape[0],)
      return images[index], temps[index], dates[index], weather[index]

  def generate_many2many_data2(self, time_step, images, temps, dates, weather):
      img_len = images.shape[0]
      train_x = []
      train_y = []
      train_d = []
      train_weather = []
      for i in range(0, img_len - time_step, time_step):   # non-overlap
          train_x.append(images[i : i + time_step+1][:])
          train_y.append(temps[i : i + time_step+1][:])
          # train_y.append(np.concatenate((temps[i : i + time_step][:], depths[i : i + time_step][:]), axis=1))
          train_d.append(dates[i : i + time_step+1][:])
          train_weather.append(weather[i : i + time_step+1][:])
      train_x = np.array(train_x)
      train_y = np.array(train_y)
      train_d = np.array(train_d)
      train_weather=np.array(train_weather)

      return train_x, train_y, train_d, train_weather

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

  def data_split(self, t_x,t_y,t_d,weather,phase,mask_frac):
      if phase=="train" or phase=="pre_train":
        train_x1 = t_x[:round(t_x.shape[0] * 0.375)]
        train_y1 = t_y[:round(t_y.shape[0] * 0.375)]
        train_d1 = t_d[:round(t_d.shape[0] * 0.375)]
        train_x2 = t_x[round(t_x.shape[0] * 0.875):]
        train_y2 = t_y[round(t_y.shape[0] * 0.875):]
        train_d2 = t_d[round(t_d.shape[0] * 0.875):]
        train_w1 = weather[:round(weather.shape[0] * 0.375)]
        train_w2 = weather[round(weather.shape[0] * 0.875):]

        imgs = np.concatenate((train_x1, train_x2), axis=0)
        labels = np.concatenate((train_y1, train_y2), axis=0)
        dates = np.concatenate((train_d1, train_d2), axis=0)
        weather = np.concatenate((train_w1, train_w2), axis=0)

        if mask_frac==-1:
            # mask sequentially
            labels[:round(t_y.shape[0] * 0.375),:,:] = [None]
        else:
            # mask randomly
            random_indices = np.random.choice(labels.size, size=int(labels.size * mask_frac), replace=False)
            row_indices, col_indices = random_indices//labels.shape[1], random_indices%labels.shape[1]
            labels[row_indices, col_indices,:]=[None]

      elif phase=="val":
        imgs = t_x[round(t_x.shape[0] * 0.675):round(t_x.shape[0] * 0.875)]
        labels = t_y[round(t_y.shape[0] * 0.675):round(t_x.shape[0] * 0.875)]
        dates = t_d[round(t_d.shape[0] * 0.675):round(t_x.shape[0] * 0.875)]
        weather = weather[round(t_d.shape[0] * 0.675):round(t_x.shape[0] * 0.875)]
      else:
        imgs = t_x[round(t_x.shape[0] * 0.375):round(t_x.shape[0] * 0.675)]
        labels = t_y[round(t_y.shape[0] * 0.375):round(t_x.shape[0] * 0.675)]
        dates = t_d[round(t_d.shape[0] * 0.375):round(t_x.shape[0] * 0.675)]
        weather=weather[round(t_d.shape[0] * 0.375):round(t_x.shape[0] * 0.675)]
      return imgs, labels, dates,weather

  def __getitem__(self, id):
    # print(id)
    imgs=torch.zeros((self.len_seq+1,3,self.image_size,self.image_size),dtype=torch.float32)
    # imgs=torch.zeros((self.len_seq,3,360,360),dtype=torch.float32)
    t=torch.zeros((self.len_seq+1,1),dtype=torch.float32)
    lbl=torch.zeros((self.len_seq+1,1),dtype=torch.float32)
    ids=np.zeros((self.len_seq+1,1))
    for i in range(self.len_seq+1):
      cur_path,url=self.images[id][i][0],self.images[id][i][1]
      img_cur=self._loadimage(cur_path,url)
      if self.transform is not None:
        img_cur=self.transform(img_cur) # The pixel values of the tensor are of type float32 and range from 0 to 1 (transforms.ToTensor() method includes normalization as part of its functionality, by dividing the pixel values by 255 to rescale them to the range [0, 1]).
      imgs[i,...]=img_cur
      t[i,...]=torch.tensor([self.dates[id][i][0]])
      lbl[i,...]=torch.tensor([self.labels[id][i][0]])
      w = np.stack([self.weather[id][i][:] for i in range(self.weather[id].shape[0])])
      w = torch.tensor(w)
      ids[i,...]=self.images[id][i][2]

#     return imgs.squeeze(), t, torch.from_numpy(self.labels[id].astype(np.float32)), w, torch.from_numpy(ids)
      images_nxt = imgs[1:]
      t_nxt = t[1:]
      label_nxt = lbl[1:]
      weather_nxt = w[1:]

      images_X = imgs[:-1]
      t_X = t[:-1]
      label_X = lbl[:-1]
      weather_X = w[:-1]
    return {"img": images_nxt, "mixed": (images_X, label_X.reshape(self.batch_size,self.l_seq,-1), weather_X.reshape(self.batch_size,self.l_seq,-1), t_nxt)}

  def __len__(self):
      return self.images.shape[0]