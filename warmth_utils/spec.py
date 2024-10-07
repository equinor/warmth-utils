
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Extra, confloat

class PData(BaseModel):
    class Config:
        extra = Extra.allow

    Name: str
    Description: Optional[str] = None
    crsName: str

    wkt: str
    epsg: int
    crsId: str
class GeomintRecordInput(BaseModel):
    class Config:
        extra = Extra.forbid

    id: str
    version: float

class Porosity(BaseModel):
    exponent: float
    initial: float


class Conductivity(BaseModel):
    porosityDependent: bool
    temperatureDependent: bool
    value: float


class HeatCapacity(BaseModel):
    porosityDependent: bool
    temperatureDependent: bool
    value: float
class ObservationData(BaseModel):
    burialDepth: float
    value: float
    observationError: float
class ObservationDataMoho(BaseModel):
    value: float
    observationError: float

class FaciesMapping(BaseModel):
    lithologyValue: str
    mapValue: int


class Sedimentsproperties(BaseModel):
    faciesMappings: List[FaciesMapping]
    faciesMap: str
    type: Literal["class"]


class VshStackItem(BaseModel):
    lithologyValue_1: str
    lithologyValue_0: str
    vshMap: str



class Sedimentsproperties1(BaseModel):
    vshStack: List[VshStackItem]
    type: Literal["vsh"]


class Sedimentsproperties2(BaseModel):
    lithologyValue_1: str
    lithologyValue_0: str
    lithCube: str
    type: Literal["cube"]

class GeomintFullSedimentaryModel(BaseModel):
    class Config:
        extra = Extra.allow

    name: str
    type: Literal['type','class','vsh']
    sedimentsproperties: Union[
        Sedimentsproperties, Sedimentsproperties1, Sedimentsproperties2
    ]
class SourceRock(BaseModel):
    sedimentsproperties: Optional[GeomintFullSedimentaryModel] = None
    thickness: Optional[Union[float, str]] = None
    HydrogenIndexInitial: Union[float, str]
    tocInitial: Union[float, str]
    kinetics: str
    name: str

class FrameworkMapping(BaseModel):
    sourceRockPos: Optional[Literal['top','middle','base']] = None
    sourceRock: Optional[SourceRock] = None
    sedimentaryModel: GeomintFullSedimentaryModel
    age: float


class InitialCondition(BaseModel):
    lithosphereThickness: confloat(ge=5000.0, le=200000.0)
    crustalThickness: confloat(ge=5000.0, le=80000.0)



class LithosphereProperties(BaseModel):
    lithosphericMantleConductivity: confloat(ge=1.0, le=3.0)
    lithosphericMantleDensity: confloat(ge=2000.0, le=5000.0)
    crustConductivity: confloat(ge=1.0, le=3.0)
    crustRHP: confloat(ge=0.0, le=3.0)
    crustDensity: confloat(ge=2000.0, le=4000.0)


class RiftEvent(BaseModel):
    end: int
    start: int


class Domain(BaseModel):
    initialCondition: InitialCondition
    lithosphereProperties: LithosphereProperties
    riftEvents: List[RiftEvent]
    extent: List[List[float]]


class TectonicModel(BaseModel):
    domains: List[Domain]
    name: str


class Geometry(BaseModel):
    top_depth_map: str
    conformity: Optional[Literal['conformable','erosion']] = None
    age: float
    name: str


class Framework(BaseModel):
    geometries: List[Geometry]
    name: str
class Model(BaseModel):
    frameworkMappings: List[FrameworkMapping]
    paleoGeometry: Optional[GeomintRecordInput] = None
    tectonicModel: TectonicModel
    framework: Framework
    name: str
    version: float
    id: str
    aoi: list[Tuple[float, float]]
    inc: Optional[int] = None

class WellboreObservations(BaseModel):
    crustalThickness: Optional[ObservationDataMoho] = None
    vitrinite: Optional[List[ObservationData]] = None
    temperature: Optional[List[ObservationData]] = None
    bias: Optional[float] = None
    y: float
    x: float

class Map(BaseModel):
    id: str
    RDDMS: List[str]
class Migri(BaseModel):
    Vss: float
    Vig: float
    Vev: float
    Vom: float
    Vcp: float
    Vca: float
    Vsh: float
class GeomintLithology(BaseModel):
    class Config:
        extra = Extra.allow

    name: str
    density: float
    porosity: Porosity
    conductivity: Conductivity
    radiogenicHeat: float
    heatCapacity: HeatCapacity
    migri: Migri
    id: str
class GeomintFullModel(BaseModel):
    class Config:
        extra = Extra.allow

    project: PData
    model: Model
    lithologies: List[GeomintLithology]
    maps: List[Map]
    pauseAfterCalibration: Optional[bool] = None
    observations: Optional[Dict[str,WellboreObservations]] = None
    uwi: Optional[str] = None



