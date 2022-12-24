from fastapi import APIRouter

from router import CanHo_ChungCu_Train
from router import Dat_Train
from router import Nha_Train

from Json import Data1 as data1
from Json import Data2 as data2
from Json import Data3 as data3


router = APIRouter()


# Get predict of can ho - chung cu
@router.post("/flat/predict")
async def getPredict1(data: data1.CanHo_Chungcu):

    DataPredict = data1.CanHo_Chungcu()

    DataPredict.Year = data.Year
    DataPredict.Area = data.Area
    DataPredict.Floor = data.Floor
    DataPredict.BedRoom = data.BedRoom
    DataPredict.Toilet = data.Toilet

    DataPredict.NameProject = data.NameProject
    DataPredict.District = data.District
    DataPredict.Ward = data.Ward
    DataPredict.Street = data.Street
    DataPredict.HomeOrientation = data.HomeOrientation
    DataPredict.Interior = data.Interior
    DataPredict.Legal = data.Legal
    DataPredict.View = data.View
    DataPredict.BalconyOrientation = data.BalconyOrientation
    DataPredict.Special = data.Special

    _data = dict()
    _data.update({  'Năm':int(DataPredict.Year),
                    'Diện tích - m2':float(DataPredict.Area),
                    'Số tầng':int(DataPredict.Floor),
                    'Số phòng ngủ':int(DataPredict.BedRoom),
                    'Số toilet':int(DataPredict.Toilet) })

    _data.update({  'Tên chung cư':str(DataPredict.NameProject),
                    'Quận':str(DataPredict.District),
                    'Phường':str(DataPredict.Ward),
                    'Đường':str(DataPredict.Street),
                    'Hướng nhà':str(DataPredict.HomeOrientation),
                    'Nội thất':str(DataPredict.Interior),
                    'Pháp lý':str(DataPredict.Legal),
                    'View':str(DataPredict.View),
                    'Hướng ban công':str(DataPredict.BalconyOrientation),
                    'Đặc trưng':str(DataPredict.Special) })

    Predict_Result = CanHo_ChungCu_Train.predict_input_user(_data)

    return Predict_Result.tolist()[0]

# Get predict of can ho - chung cu
@router.post("/house/predict")
async def getPredict2(data: data2.House):

    DataPredict = data2.House()

    DataPredict.Year = data.Year
    DataPredict.Area = data.Area
    DataPredict.Floor = data.Floor
    DataPredict.Facade = data.Facade
    DataPredict.AccessRoad = data.AccessRoad
    DataPredict.BedRoom = data.BedRoom
    DataPredict.Toilet = data.Toilet

    DataPredict.District = data.District
    DataPredict.Ward = data.Ward
    DataPredict.Street = data.Street
    DataPredict.HomeOrientation = data.HomeOrientation
    DataPredict.BalconyOrientation = data.BalconyOrientation
    DataPredict.Interior = data.Interior
    DataPredict.Legal = data.Legal

    _data = dict()
    _data.update({  'Year':int(DataPredict.Year),
                    'Area':float(DataPredict.Area),
                    'Floor':int(DataPredict.Floor),
                    'Facade':int(DataPredict.Facade),
                    'AccessRoad':int(DataPredict.AccessRoad),
                    'BedRoom':int(DataPredict.BedRoom),
                    'Toilet':int(DataPredict.Toilet) })

    _data.update({  'District':str(DataPredict.District),
                    'Ward':str(DataPredict.Ward),
                    'Street':str(DataPredict.Street),
                    'HomeOrientation':str(DataPredict.HomeOrientation),
                    'BalconyOrientation':str(DataPredict.BalconyOrientation),
                    'Interior':str(DataPredict.Interior),
                    'Legal':str(DataPredict.Legal), })

    Predict_Result = Nha_Train.predict_input_user(_data)

    return Predict_Result.tolist()[0]

# Get predict of can ho - chung cu
@router.post("/plot/predict")
async def getPredict3(data: data3.Plot):

    DataPredict = data3.Plot()

    DataPredict.Year = data.Year
    DataPredict.Area = data.Area
    DataPredict.Facade = data.Facade
    DataPredict.AccessRoad = data.AccessRoad
    DataPredict.GPXD = data.GPXD

    DataPredict.District = data.District
    DataPredict.Ward = data.Ward
    DataPredict.Street = data.Street
    DataPredict.Legal = data.Legal

    _data = dict()
    _data.update({  'Year':int(DataPredict.Year),
                    'Area':float(DataPredict.Area),
                    'Facade':int(DataPredict.Facade),
                    'AccessRoad':int(DataPredict.AccessRoad),
                    'GPXD':int(DataPredict.GPXD), })

    _data.update({  'District':str(DataPredict.District),
                    'Ward':str(DataPredict.Ward),
                    'Street':str(DataPredict.Street),
                    'Legal':str(DataPredict.Legal), })

    Predict_Result = Dat_Train.predict_input_user(_data)

    return Predict_Result.tolist()[0]
