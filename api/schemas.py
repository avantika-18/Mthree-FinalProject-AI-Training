from pydantic import BaseModel, field_validator
from typing import Optional

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    rooms_per_household: Optional[float] = None  # Derived if not provided

    @field_validator('MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude')
    @classmethod
    def validate_ranges(cls, v, field):
        ranges = {
            'MedInc': (0, 15),  # Median income 0-15
            'HouseAge': (0, 52),
            'AveRooms': (0, 50),
            'AveBedrms': (0, 10),
            'Population': (0, 40000),
            'AveOccup': (0, 10),
            'Latitude': (32, 42),  # CA range
            'Longitude': (-125, -114)
        }
        min_val, max_val = ranges[field.name]
        if not min_val <= v <= max_val:
            raise ValueError(f"{field.name} out of realistic range: {min_val}-{max_val}")
        return v