
from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "configs", "isos.yaml")


@dataclass(frozen=True)
class LoadForecastConfig:
    mode: str                            
    column: str | None = None
    zonal_columns: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LmpComponentsConfig:
    energy: str | None = None
    congestion: str | None = None
    loss: str | None = None


@dataclass(frozen=True)
class IsoConfig:
    name: str
    tz: str
    locations: tuple[str, ...]
    load_forecast: LoadForecastConfig
    lmp_components: LmpComponentsConfig
    lmp_column: str = "LMP"
                                                                                              
    gate_local_hour: float = 0.0


@dataclass(frozen=True)
class SplitsConfig:
    train_end: str
    val_end: str


@dataclass(frozen=True)
class GasPriceConfig:
    
    mode: str = "none"                              
    lag_days: int = 2


@dataclass(frozen=True)
class ProjectConfig:
    isos: tuple[IsoConfig, ...]
    splits: SplitsConfig
    gas_price: GasPriceConfig = field(default_factory=GasPriceConfig)

    def iso(self, name: str) -> IsoConfig:
        for c in self.isos:
            if c.name == name:
                return c
        raise KeyError(f"no config for ISO {name!r}")

    @property
    def iso_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.isos)


                                                                    
FOLD_SPLITS: dict[str, SplitsConfig] = {
    "A": SplitsConfig(train_end="2021-12-31", val_end="2022-12-31"),               
    "B": SplitsConfig(train_end="2022-12-31", val_end="2023-12-31"),               
}


def with_fold(cfg: ProjectConfig, fold: str | None) -> ProjectConfig:
    
    import dataclasses
    if fold is None or fold == "C":
        return cfg
    if fold not in FOLD_SPLITS:
        raise ValueError(f"unknown fold {fold!r}, expected one of {sorted(FOLD_SPLITS)} or 'C'")
    return dataclasses.replace(cfg, splits=FOLD_SPLITS[fold])


def load_config(path: str = CONFIG_PATH) -> ProjectConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    isos = []
    for name, spec in raw["isos"].items():
        lf = spec.get("load_forecast", {"mode": "none"})
        lc = spec.get("lmp_components", {})
        isos.append(IsoConfig(
            name=name,
            tz=spec["tz"],
            locations=tuple(spec["locations"]),
            lmp_column=spec.get("lmp_column", "LMP"),
            gate_local_hour=spec.get("gate_local_hour", 0.0),
            load_forecast=LoadForecastConfig(
                mode=lf["mode"],
                column=lf.get("column"),
                zonal_columns=tuple(lf.get("zonal_columns", [])),
            ),
            lmp_components=LmpComponentsConfig(
                energy=lc.get("energy"), congestion=lc.get("congestion"), loss=lc.get("loss"),
            ),
        ))

    gp = raw.get("gas_price", {"mode": "none"})
    return ProjectConfig(
        isos=tuple(isos),
        splits=SplitsConfig(**raw["splits"]),
        gas_price=GasPriceConfig(mode=gp.get("mode", "none"), lag_days=gp.get("lag_days", 2)),
    )
