"""sfr_p01 module"""

# pylint: disable=R0801
from importlib.resources import files
from os import PathLike
from typing import Iterator, Union

import flopy as fp
import numpy as np
import pandas as pd

from . import data


class SFRP01:
    """Streamflow Routing example

    Parameters
    ----------
    exe_name : str or PathLike
    sim_ws : str or PathLike

    Attributes
    ----------
    sim_name : str
        Simulation name is `gwf_sfr_p01`.
    simulation : flopy.mf6.MFSimulation
    model : flopy.mf6.ModflowGwf
    """

    sim_name = "gwf_sfr_p01"

    def __init__(self, exe_name: Union[str, PathLike], sim_ws: Union[str, PathLike]):
        # setup simulation
        self._simulation = fp.mf6.MFSimulation(
            sim_name=self.sim_name, version="mf6", exe_name=exe_name, sim_ws=sim_ws
        )
        self._add_sim_tdis()
        self._add_sim_ims()

        # setup model
        self._model = fp.mf6.ModflowGwf(
            simulation=self._simulation,
            modelname=self.sim_name,
            model_nam_file=f"{self.sim_name}.nam",
            save_flows=True,
            newtonoptions="NEWTON",
        )
        self._add_model_dis()
        self._add_model_npf()
        self._add_model_sto()
        self._add_model_ic()
        self._add_model_ghb()
        self._add_model_wel()
        self._add_model_rcha()
        self._add_model_evta()
        self._add_model_sfr()
        self._add_model_oc()

    @property
    def simulation(self):
        """Returns MODFLOW 6 simulation object."""
        return self._simulation

    @property
    def model(self):
        """Returns groundwater flow model object."""
        return self._model

    def check_simulation(self):
        """Check simulation for errors and warnings."""
        self._simulation.check()

    def write_simulation(self):
        """Write MODFLOW simulation files."""
        self._simulation.write_simulation()

    def run_simulation(self):
        """Run MODFLOW simulation.

        Raises
        ------
        AssertionError
            If MODFLOW does not terminate normally.
        """
        success, _ = self._simulation.run_simulation()
        assert success, "MODFLOW did not terminate normally!"

    def _add_sim_tdis(self):
        """Add Temporal Discretization to simulation."""
        try:
            fp.mf6.ModflowTdis(
                simulation=getattr(self, "_simulation"),
                time_units="seconds",
                nper=3,
                perioddata=[(0.0, 1, 1.0), (1.57788e9, 50, 1.1), (1.57788e9, 50, 1.1)],
                filename=f"{self.sim_name}.tdis",
                pname="tdis",
            )
        except AttributeError:
            # simulation doesn't exist, do nothing
            pass

    def _add_sim_ims(self):
        """Add Iterative Model Solution to simulation."""
        try:
            fp.mf6.ModflowIms(
                simulation=getattr(self, "_simulation"),
                print_option="summary",
                outer_dvclose=1e-6,
                outer_maximum=100,
                inner_maximum=50,
                inner_dvclose=1e-6,
                rcloserecord=[1e-6, "strict"],
                linear_acceleration="bicgstab",
                filename=f"{self.sim_name}.ims",
                pname="ims",
            )
        except AttributeError:
            # simulation doesn't exist, do nothing
            pass

    def _add_model_dis(self):
        """Add Structured Discretization (DIS) to model."""
        try:
            with np.load(files(data) / "dis.npz") as dis_data:
                fp.mf6.ModflowGwfdis(
                    model=getattr(self, "_model"),
                    length_units="feet",
                    nlay=1,
                    nrow=15,
                    ncol=10,
                    delr=5000.0,
                    delc=5000.0,
                    top=dis_data["top"],
                    botm=dis_data["botm"],
                    idomain=dis_data["idomain"],
                    filename=f"{self.sim_name}.dis",
                    pname="dis",
                )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_npf(self):
        """Add Node Property Flow (NPF) package to model."""
        try:
            with np.load(files(data) / "npf.npz") as npf_data:
                fp.mf6.ModflowGwfnpf(
                    model=getattr(self, "_model"),
                    save_specific_discharge=True,
                    # unconfined
                    icelltype=1,
                    k=npf_data["k"],
                    filename=f"{self.sim_name}.npf",
                    pname="npf",
                )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_sto(self):
        """Add Storage (STO) package to model."""
        try:
            with np.load(files(data) / "sto.npz") as sto_data:
                fp.mf6.ModflowGwfsto(
                    model=getattr(self, "_model"),
                    iconvert=1,
                    ss=1e-6,
                    sy=sto_data["sy"],
                    steady_state={0: True},
                    transient={1: True},
                    filename=f"{self.sim_name}.sto",
                    pname="sto",
                )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_ic(self):
        """Add Initial Conditions to model."""
        try:
            fp.mf6.ModflowGwfic(
                model=getattr(self, "_model"),
                strt=1050.0,
                filename=f"{self.sim_name}.ic",
                pname="ic",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_ghb(self):
        """Add General-Head Boundary (GHB) package to model."""
        try:
            fp.mf6.ModflowGwfghb(
                model=getattr(self, "_model"),
                maxbound=2,
                stress_period_data=[
                    ((0, 12, 0), 988.0, 0.038),
                    ((0, 13, 8), 1045.0, 0.038),
                ],
                filename=f"{self.sim_name}.ghb",
                pname="ghb",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_wel(self):
        """Add Well (WEL) package to model."""

        def _make_wel_data() -> Iterator[tuple[tuple[int, int, int], float]]:
            """Make stress period data for well package.

            Yields
            ------
            (layer, row, col), q
            """
            for row in range(5, 10):
                for col in range(3, 5):
                    yield (0, row, col), -10.0

        try:
            fp.mf6.ModflowGwfwel(
                model=getattr(self, "_model"),
                maxbound=10,
                stress_period_data={1: list(_make_wel_data()), 2: []},
                filename=f"{self.sim_name}.wel",
                pname="wel",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_rcha(self):
        """Add array-based Recharge (RCHA) package to model."""
        try:
            with np.load(files(data) / "rcha.npz") as rcha_data:
                fp.mf6.ModflowGwfrcha(
                    model=getattr(self, "_model"),
                    recharge=rcha_data["recharge"],
                    filename=f"{self.sim_name}.rcha",
                    pname="rcha",
                )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_evta(self):
        """Add array-based Evapotranspiration (EVTA) package to model."""
        try:
            with np.load(files(data) / "evta.npz") as evta_data:
                fp.mf6.ModflowGwfevta(
                    model=getattr(self, "_model"),
                    surface=evta_data["surface"],
                    rate=9.5e-8,
                    depth=15.0,
                    filename=f"{self.sim_name}.evta",
                    pname="evta",
                )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_sfr(self):
        """Add Streamflow Routing (SFR) package to model."""
        try:
            packagedata_df = pd.read_parquet(files(data) / "sfr_packagedata.parquet")
            packagedata_df["cellid"] = packagedata_df["cellid"].apply(tuple)
            # packagedata_df.to_records(index=False)

            connectiondata_adj = np.load(files(data) / "sfr_connectiondata.npy")
            # construct tuple-based graph
            len_ic = packagedata_df["ncon"].max()
            connectiondata = []
            for reach in range(36):
                edges = sorted(
                    [
                        # upstream from current reach
                        *np.argwhere(connectiondata_adj[:, reach]).reshape(-1),
                        # downstream from current reach
                        *(np.argwhere(connectiondata_adj[reach, :]).reshape(-1) * -1),
                    ],
                    key=abs,
                )  # sort by absolute value
                if len(edges) < len_ic:
                    edges.extend([None] * (len_ic - len(edges)))
                connectiondata.append(tuple([reach, *edges]))

            fp.mf6.ModflowGwfsfr(
                model=getattr(self, "_model"),
                observations={
                    f"{self.sim_name}.sfr.obs.csv": [
                        ("r01_stage", "stage", "4", None),
                        ("r02_stage", "stage", "15", None),
                        ("r03_stage", "stage", "27", None),
                        ("r04_stage", "stage", "36", None),
                        ("r01_flow", "downstream-flow", "4", None),
                        ("r02_flow", "downstream-flow", "15", None),
                        ("r03_flow", "downstream-flow", "27", None),
                        ("r04_flow", "downstream-flow", "36", None),
                    ]
                },
                unit_conversion=1.486,
                nreaches=36,
                packagedata=packagedata_df.to_records(index=False),
                connectiondata=connectiondata,
                # list of (ifno, idv, iconr, cprior)
                diversions=[(3, 0, 9, "upto")],
                perioddata=[
                    (0, "inflow", "25.0", None),
                    (15, "inflow", "10.0", None),
                    (27, "inflow", "150.0", None),
                    (3, "diversion", 0, 10.0),
                    (9, "status", "simple", None),
                    (10, "status", "simple", None),
                    (11, "status", "simple", None),
                    (12, "status", "simple", None),
                    (13, "status", "simple", None),
                    (14, "status", "simple", None),
                    (9, "stage", "1075.545", None),
                    (10, "stage", "1072.636", None),
                    (11, "stage", "1069.873", None),
                    (12, "stage", "1066.819", None),
                    (13, "stage", "1063.619", None),
                    (14, "stage", "1061.581", None),
                ],
                filename=f"{self.sim_name}.sfr",
                pname="sfr",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_oc(self):
        """Add Output Control to model."""
        try:
            fp.mf6.ModflowGwfoc(
                model=getattr(self, "_model"),
                budget_filerecord=[f"{self.sim_name}.cbc"],
                head_filerecord=[f"{self.sim_name}.hds"],
                saverecord=[("head", "all"), ("budget", "all")],
                filename=f"{self.sim_name}.oc",
                pname="oc",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass
