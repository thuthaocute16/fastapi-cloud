from typing import Union
from pydantic import BaseModel


class CanHo_Chungcu(BaseModel) : 
    Year: Union[int, None] = None
    Area: Union[float, None] = None
    Floor: Union[int, None] = None
    BedRoom: Union[int, None] = None
    Toilet: Union[int, None] = None

    NameProject: Union[str, None] = None
    District: Union[str, None] = None
    Ward: Union[str, None] = None
    Street: Union[str, None] = None
    HomeOrientation: Union[str, None] = None
    Interior: Union[str, None] = None
    Legal: Union[str, None] = None
    View: Union[str, None] = None
    BalconyOrientation: Union[str, None] = None
    Special: Union[str, None] = None 