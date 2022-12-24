
from typing import Union
from pydantic import BaseModel


class House(BaseModel) : 
    Year: Union[int, None] = None
    Area: Union[float, None] = None
    Floor: Union[int, None] = None
    Facade: Union[float, None] = None
    AccessRoad: Union[float, None] = None
    BedRoom: Union[int, None] = None
    Toilet: Union[int, None] = None

    District: Union[str, None] = None
    Ward: Union[float, None] = None
    Street: Union[float, None] = None
    HomeOrientation: Union[float, None] = None
    BalconyOrientation: Union[float, None] = None
    Interior: Union[float, None] = None
    Legal: Union[float, None] = None