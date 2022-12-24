from typing import Union
from pydantic import BaseModel


class Plot(BaseModel) : 
    Year: Union[int, None] = None
    Area: Union[float, None] = None
    Facade: Union[float, None] = None
    AccessRoad: Union[float, None] = None
    GPXD: Union[int, None] = None

    District: Union[str, None] = None
    Ward: Union[str, None] = None
    Street: Union[str, None] = None
    Legal: Union[str, None] = None