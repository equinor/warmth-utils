
from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Extra, confloat
class GeomintsPolygon(BaseModel):
    __root__: List[List[float]]
class PData(BaseModel):
    class Config:
        extra = Extra.allow

    Name: str
    Description: Optional[str] = None
    crsName: str

    wkt: str
    epsg: float
    crsId: str
class GeomintRecordInput(BaseModel):
    class Config:
        extra = Extra.forbid

    id: str
    version: float

class Porosity(BaseModel):
    exponent: float
    initial: float


class PorosityDependent(Enum):
    boolean_True = True


class TemperatureDependent(Enum):
    boolean_True = True


class Conductivity(BaseModel):
    porosityDependent: PorosityDependent
    temperatureDependent: TemperatureDependent
    value: float


class PorosityDependent1(Enum):
    boolean_False = False


class TemperatureDependent1(Enum):
    boolean_False = False


class HeatCapacity(BaseModel):
    porosityDependent: PorosityDependent1
    temperatureDependent: TemperatureDependent1
    value: float
class ObservationData(BaseModel):
    burialDepth: float
    value: float
    observationError: float
class Type4(Enum):
    class_ = 'class'
    vsh = 'vsh'
    cube = 'cube'


class FaciesMapping(BaseModel):
    lithologyValue: str
    mapValue: int


class Type5(Enum):
    class_ = 'class'


class Sedimentsproperties(BaseModel):
    faciesMappings: List[FaciesMapping]
    faciesMap: str
    type: Type5


class VshStackItem(BaseModel):
    lithologyValue_1: str
    lithologyValue_0: str
    vshMap: str


class Type6(Enum):
    vsh = 'vsh'


class Sedimentsproperties1(BaseModel):
    vshStack: List[VshStackItem]
    type: Type6


class Type7(Enum):
    cube = 'cube'


class Sedimentsproperties2(BaseModel):
    lithologyValue_1: str
    lithologyValue_0: str
    lithCube: str
    type: Type7

class GeomintFullSedimentaryModel(BaseModel):
    class Config:
        extra = Extra.allow

    name: str
    type: Type4
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
class SourceRockPos(Enum):
    top = 'top'
    middle = 'middle'
    base = 'base'
class FrameworkMapping(BaseModel):
    sourceRockPos: Optional[SourceRockPos] = None
    sourceRock: Optional[SourceRock] = None
    sedimentaryModel: GeomintFullSedimentaryModel
    age: float
class LithosphereThickness(BaseModel):
    __root__: confloat(ge=5000.0, le=200000.0)


class LithosphereThickness1(BaseModel):
    __root__: str


class CrustalThickness(BaseModel):
    __root__: confloat(ge=5000.0, le=80000.0)


class CrustalThickness1(BaseModel):
    __root__: str


class InitialCondition(BaseModel):
    lithosphereThickness: Union[LithosphereThickness, LithosphereThickness1]
    crustalThickness: Union[CrustalThickness, CrustalThickness1]


class CrustConductivity(BaseModel):
    __root__: confloat(ge=1.0, le=3.0)


class CrustConductivity1(BaseModel):
    __root__: str


class CrustRHP(BaseModel):
    __root__: confloat(ge=0.0, le=3.0)


class CrustRHP1(BaseModel):
    __root__: str


class CrustDensity(BaseModel):
    __root__: confloat(ge=2000.0, le=4000.0)


class CrustDensity1(BaseModel):
    __root__: str


class LithosphereProperties(BaseModel):
    lithosphericMantleConductivity: confloat(ge=1.0, le=3.0)
    lithosphericMantleDensity: confloat(ge=2000.0, le=5000.0)
    crustConductivity: Union[CrustConductivity, CrustConductivity1]
    crustRHP: Union[CrustRHP, CrustRHP1]
    crustDensity: Union[CrustDensity, CrustDensity1]


class RiftEvent(BaseModel):
    end: int
    start: int


class Domain(BaseModel):
    initialCondition: InitialCondition
    lithosphereProperties: LithosphereProperties
    riftEvents: List[RiftEvent]
    extent: GeomintsPolygon


class TectonicModel(BaseModel):
    domains: List[Domain]
    name: str

class Conformity(Enum):
    conformable = 'conformable'
    erosion = 'erosion'


class Geometry(BaseModel):
    top_depth_map: str
    conformity: Optional[Conformity] = None
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
class FieldXNumberYNumberBias63NumberTemperature63ObservationDataArrayVitrinite63ObservationDataArrayCrustalThickness6358ValueNumberObservationErrorNumber(
    BaseModel
):
    crustalThickness: Optional[CrustalThickness] = None
    vitrinite: Optional[List[ObservationData]] = None
    temperature: Optional[List[ObservationData]] = None
    bias: Optional[float] = None
    y: float
    x: float
class FieldXNumberYNumberBias63NumberTemperature63ObservationDataArrayVitrinite63ObservationDataArrayCrustalThickness6358ValueNumberObservationErrorNumberModel(
    BaseModel
):
    __root__: Optional[
        Dict[
            str,
            FieldXNumberYNumberBias63NumberTemperature63ObservationDataArrayVitrinite63ObservationDataArrayCrustalThickness6358ValueNumberObservationErrorNumber,
        ]
    ] = None

class WellboreObservations(BaseModel):
    __root__: (
        FieldXNumberYNumberBias63NumberTemperature63ObservationDataArrayVitrinite63ObservationDataArrayCrustalThickness6358ValueNumberObservationErrorNumberModel
    )
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
class GeomintFullModel(BaseModel):
    class Config:
        extra = Extra.allow

    project: PData
    model: Model
    lithologies: List[GeomintLithology]
    maps: List[Map]
    pauseAfterCalibration: Optional[bool] = None
    observations: Optional[WellboreObservations] = None



