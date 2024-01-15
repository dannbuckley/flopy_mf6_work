"""advtidal module"""
# pylint: disable=R0801
from importlib.resources import files, as_file
from os import PathLike
from typing import Iterator, Union

import flopy as fp
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd

from . import data


class Tidal:
    """Tidal example
    (see mf6examples.pdf, pages 3-1 through 3-4)

    Parameters
    ----------
    exe_name : str or PathLike
    sim_ws : str or PathLike

    Attributes
    ----------
    sim_name : str
        Simulation name is `gwf_tidal`.
    simulation : flopy.mf6.MFSimulation
    model : flopy.mf6.ModflowGwf
    """

    sim_name = "gwf_tidal"

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
        )
        self._add_model_dis()
        self._add_model_npf()
        self._add_model_ic()
        self._add_model_sto()
        self._add_model_ghb()
        self._add_model_wel()
        self._add_model_riv()
        self._add_model_rch()
        self._add_model_evt()
        self._add_model_oc()
        self._add_model_obs()

    @property
    def simulation(self):
        """Get simulation object for this example."""
        return self._simulation

    @property
    def model(self):
        """Get groundwater flow model object for this example."""
        return self._model

    def check_simulation(self):
        """Check MODFLOW simulation for errors and warnings."""
        self._simulation.check()

    def write_simulation(self):
        """Write files for MODFLOW simulation."""
        self._simulation.write_simulation()

    def run_simulation(self):
        """Run MODFLOW simulation.

        Raises
        ------
        AssertionError
            If MODFLOW simulation does not terminate normally.
        """
        success, _ = self._simulation.run_simulation()
        assert success, "MODFLOW did not terminate normally!"

    def plot_recharge_zones(self):
        """Generate a map-view plot of the 3 recharge zones.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(figsize=(4, 6))
        axes.set_title("Recharge Zones", fontweight="bold")
        mapview = fp.plot.PlotMapView(model=self._model, ax=axes, layer=0)
        mapview.plot_ibound()
        mapview.plot_grid()

        # plot recharge zones
        mapview.plot_bc(
            package=self._model.get_package("rch-zone_1"), color="tab:red", alpha=0.4
        )
        mapview.plot_bc(
            package=self._model.get_package("rch-zone_2"), color="tab:blue", alpha=0.4
        )
        mapview.plot_bc(
            package=self._model.get_package("rch-zone_3"), color="tab:green", alpha=0.4
        )

        # add legend
        axes.legend(
            handles=[
                Patch(color="tab:red", alpha=0.4, label="Zone 1"),
                Patch(color="tab:blue", alpha=0.4, label="Zone 2"),
                Patch(color="tab:green", alpha=0.4, label="Zone 3"),
            ],
            loc="lower left",
        )

        # return completed figure to caller
        return fig

    def plot_mapview_unconfined_aq(self):
        """Generate a map-view plot of the unconfined aquifer with
        well and river boundary conditions.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(figsize=(4, 6))
        axes.set_title("Map View of Unconfined Aquifer", fontweight="bold")
        mapview = fp.plot.PlotMapView(model=self._model, ax=axes, layer=0)
        mapview.plot_ibound()
        mapview.plot_grid()

        # plot wells
        mapview.plot_bc("WEL", kper=3)
        # plot stream
        mapview.plot_bc("RIV")

        # return completed figure to caller
        return fig

    def plot_mapview_confining_unit(self):
        """Generate a map-view plot of the confining unit with
        the general-head boundary condition.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(figsize=(4, 6))
        axes.set_title("Map View of Confining Unit", fontweight="bold")
        mapview = fp.plot.PlotMapView(model=self._model, ax=axes, layer=1)
        mapview.plot_ibound()
        mapview.plot_grid()

        # plot general-head
        mapview.plot_bc("GHB")

        # return completed figure to caller
        return fig

    def plot_mapview_confined_aq(self):
        """Generate a map-view plot of the confined aquifer with
        the well and general-head boundary conditions.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(figsize=(4, 6))
        axes.set_title("Map View of Confined Aquifer", fontweight="bold")
        mapview = fp.plot.PlotMapView(model=self._model, ax=axes, layer=2)
        mapview.plot_ibound()
        mapview.plot_grid()

        # plot general-head
        mapview.plot_bc("GHB")
        # plot wells
        mapview.plot_bc("WEL", kper=3)

        # return completed figure to caller
        return fig

    def _add_sim_tdis(self):
        """Add Temporal Discretization (TDIS) package to simulation."""
        try:
            fp.mf6.ModflowTdis(
                simulation=getattr(self, "_simulation"),
                time_units="days",
                nper=4,
                # list of (perlen, nstp, tsmult)
                perioddata=[
                    # first stress period, steady-state
                    (1.0, 1, 1.0),
                    # second stress period, transient
                    (10.0, 120, 1.0),
                    # third stress period, transient
                    (10.0, 120, 1.0),
                    # fourth stress period, transient
                    (10.0, 120, 1.0),
                ],
                filename=f"{self.sim_name}.tdis",
                pname="tdis",
            )
        except AttributeError:
            # simulation doesn't exist, do nothing
            pass

    def _add_sim_ims(self):
        """Add Iterative Model Solution (IMS) to simulation."""
        try:
            fp.mf6.ModflowIms(
                simulation=getattr(self, "_simulation"),
                outer_dvclose=1e-9,
                outer_maximum=50,
                inner_maximum=100,
                inner_dvclose=1e-9,
                rcloserecord=[1e-6, "strict"],
                filename=f"{self.sim_name}.ims",
                pname="ims",
            )
        except AttributeError:
            # simulation doesn't exist, do nothing
            pass

    def _add_model_dis(self):
        """Add Structured Discretization (DIS) to model."""
        try:
            fp.mf6.ModflowGwfdis(
                model=getattr(self, "_model"),
                length_units="meters",
                nlay=3,
                nrow=15,
                ncol=10,
                delr=500.0,
                delc=500.0,
                top=50.0,
                botm=[
                    # bottom of unconfined aquifer
                    5.0,
                    # bottom of confining unit
                    -10.0,
                    # bottom of confined aquifer
                    -100.0,
                ],
                filename=f"{self.sim_name}.dis",
                pname="dis",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_npf(self):
        """Add Node Property Flow (NPF) package to model."""
        try:
            fp.mf6.ModflowGwfnpf(
                model=getattr(self, "_model"),
                cvoptions=[(True, "DEWATERED")],
                perched=True,
                save_specific_discharge=True,
                icelltype=[
                    # unconfined aquifer is unconfined/convertible
                    1,
                    # confining unit is confined
                    0,
                    # confined aquifer is confined
                    0,
                ],
                # (horizontal) hydraulic conductivity
                k=[
                    # unconfined aquifer
                    5.0,
                    # confining unit
                    0.1,
                    # confined aquifer
                    4.0,
                ],
                # (vertical) hydraulic conductivity
                k33=[
                    # unconfined aquifer
                    0.5,
                    # confining unit
                    0.005,
                    # confined aquifer
                    0.1,
                ],
                filename=f"{self.sim_name}.npf",
                pname="npf",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_ic(self):
        """Add Initial Conditions (IC) package to model."""
        try:
            fp.mf6.ModflowGwfic(
                model=getattr(self, "_model"),
                # starting head set to top of model
                strt=50.0,
                filename=f"{self.sim_name}.ic",
                pname="ic",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_sto(self):
        """Add Storage (STO) package to model."""
        try:
            fp.mf6.ModflowGwfsto(
                model=getattr(self, "_model"),
                # flag for each cell that specifies whether or not a cell is convertible
                #   for the storage calculation
                # iconvert > 0: confined storage is used when head is above cell top and
                #   and a mixed formulation of unconfined and confined storage is used when
                #   head is below cell top
                iconvert=1,
                # specific storage
                ss=1e-6,
                # specific yield
                sy=0.2,
                # indicates stress period IPER is steady-state
                steady_state=True,
                # indicates stress period IPER is transient
                transient=True,
                filename=f"{self.sim_name}.sto",
                pname="sto",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_ghb(self):
        """Add General-Head Boundary (GHB) package to model."""

        def _make_ghb_stress_data() -> (
            Iterator[tuple[tuple[int, int, int], str, float, str]]
        ):
            """Generate stress period data for general-head boundary condition.

            Yields
            ------
            ((layer, row, column), bhead, cond, boundname)
            """
            cond_dict = {1: 15.0, 2: 1500.0}
            for layer in [1, 2]:
                for row in range(15):
                    yield (
                        (layer, row, 9),
                        "tides",
                        cond_dict[layer],
                        f"estuary-l{layer + 1}",
                    )

        try:
            # load timeseries data
            with as_file(files(data).joinpath("data_ghb_ts.csv")) as csv:
                ghb_ts_df = pd.read_csv(csv)

            stress_period_data = list(_make_ghb_stress_data())
            ghb = fp.mf6.ModflowGwfghb(
                model=getattr(self, "_model"),
                boundnames=True,
                # generate observations after each time step and output to csv
                observations={
                    # list of (obsname, obstype, id, id2)
                    # obsname: observation identifier (column name in output file)
                    # obstype: flow between groundwater system and general-head boundary
                    # id: cellid or boundname
                    f"{self.sim_name}.ghb.obs.csv": [
                        ("ghb_2_6_10", "ghb", (1, 5, 9), None),
                        ("ghb_3_6_10", "ghb", (2, 5, 9), None),
                        ("estuary2", "ghb", "estuary-l2", None),
                        ("estuary3", "ghb", "estuary-l3", None),
                    ]
                },
                maxbound=len(stress_period_data),
                # list of (cellid, bhead, cond, boundname)
                # bhead: boundary head
                # cond: hydraulic conductance between aquifer cell and boundary
                # boundname: name of the general-head boundary cell
                stress_period_data=stress_period_data,
                filename=f"{self.sim_name}.ghb",
                pname="ghb",
            )

            # add timeseries data to GHB package
            fp.mf6.ModflowUtlts(
                parent_package=ghb,
                time_series_namerecord=ghb_ts_df.columns[1],
                interpolation_methodrecord="linear",
                timeseries=ghb_ts_df.to_records(index=False),
                filename=f"{self.sim_name}.ghb.ts",
                pname="ghb_ts",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_wel(self):
        """Add Well (WEL) package to model."""
        try:
            # load timeseries data
            with as_file(files(data).joinpath("data_wel_ts.csv")) as csv:
                wel_ts_df = pd.read_csv(csv)

            wel = fp.mf6.ModflowGwfwel(
                model=getattr(self, "_model"),
                boundnames=True,
                maxbound=5,
                # list of (cellid, q, boundname)
                stress_period_data={
                    # second stress period
                    1: [
                        ((0, 11, 2), -50.0, None),
                        ((2, 3, 2), "well_2_rate", "well_2"),
                        ((2, 4, 7), "well_1_rate", "well_1"),
                    ],
                    # third stress period
                    2: [
                        ((0, 2, 4), -20.0, None),
                        ((0, 11, 2), -10.0, None),
                        ((0, 13, 5), -40.0, None),
                        ((2, 3, 2), "well_2_rate", "well_2"),
                        ((2, 4, 7), "well_1_rate", "well_1"),
                    ],
                    # fourth stress period
                    3: [
                        ((0, 2, 4), -20.0, None),
                        ((0, 11, 2), -10.0, None),
                        ((0, 13, 5), -40.0, None),
                        ((2, 3, 2), "well_2_rate", "well_2"),
                        ((2, 4, 7), "well_1_rate", "well_1"),
                    ],
                },
                filename=f"{self.sim_name}.wel",
                pname="wel",
            )

            # add timeseries data to WEL package
            fp.mf6.ModflowUtlts(
                parent_package=wel,
                time_series_namerecord=wel_ts_df.columns[1:].tolist(),
                interpolation_methodrecord=["stepwise", "stepwise", "stepwise"],
                timeseries=wel_ts_df.to_records(index=False),
                filename=f"{self.sim_name}.wel.ts",
                pname="wel_ts",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_riv(self):
        """Add River (RIV) package to model."""
        try:
            # load timeseries data
            with as_file(files(data).joinpath("data_riv_ts.csv")) as csv:
                riv_ts_df = pd.read_csv(csv)

            riv = fp.mf6.ModflowGwfriv(
                model=getattr(self, "_model"),
                boundnames=True,
                maxbound=20,
                # list of (cellid, stage, cond, rbot, boundname)
                stress_period_data=[
                    ((0, 2, 0), "river_stage_1", 1001.0, 35.9, None),
                    ((0, 3, 1), "river_stage_1", 1002.0, 35.8, None),
                    ((0, 4, 2), "river_stage_1", 1003.0, 35.7, None),
                    ((0, 4, 3), "river_stage_1", 1004.0, 35.6, None),
                    ((0, 5, 4), "river_stage_1", 1005.0, 35.5, None),
                    ((0, 5, 5), "river_stage_1", 1006.0, 35.4, "riv1_c6"),
                    ((0, 5, 6), "river_stage_1", 1007.0, 35.3, "riv1_c7"),
                    ((0, 4, 7), "river_stage_1", 1008.0, 35.2, None),
                    ((0, 4, 8), "river_stage_1", 1009.0, 35.1, None),
                    ((0, 4, 9), "river_stage_1", 1010.0, 35.0, None),
                    ((0, 9, 0), "river_stage_2", 1001.0, 36.9, "riv2_upper"),
                    ((0, 8, 1), "river_stage_2", 1002.0, 36.8, "riv2_upper"),
                    ((0, 7, 2), "river_stage_2", 1003.0, 36.7, "riv2_upper"),
                    ((0, 6, 3), "river_stage_2", 1004.0, 36.6, None),
                    ((0, 6, 4), "river_stage_2", 1005.0, 36.5, None),
                    ((0, 5, 5), "river_stage_2", 1006.0, 36.4, "riv2_c6"),
                    ((0, 5, 6), "river_stage_2", 1007.0, 36.3, "riv2_c7"),
                    ((0, 6, 7), "river_stage_2", 1008.0, 36.2, None),
                    ((0, 6, 8), "river_stage_2", 1009.0, 36.1, None),
                    ((0, 6, 9), "river_stage_2", 1010.0, 36.0, None),
                ],
                filename=f"{self.sim_name}.riv",
                pname="riv",
            )

            # add timeseries data to RIV package
            fp.mf6.ModflowUtlts(
                parent_package=riv,
                time_series_namerecord=riv_ts_df.columns[1:].tolist(),
                interpolation_methodrecord=["linear", "stepwise"],
                timeseries=riv_ts_df.to_records(index=False),
                filename=f"{self.sim_name}.riv.ts",
                pname="riv_ts",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_rch(self):
        """Add Recharge (RCH) package to model."""

        def _make_rch_stress_data(
            zone: int,
        ) -> Iterator[tuple[tuple[int, int, int], str, float, None]]:
            """Generate stress period data for recharge boundary condition.

            Parameters
            ----------
            zone : int
                Recharge zone number.

            Yields
            ------
            ((layer, row, column), recharge, multiplier, boundname)
            """
            overlap = {
                (0, 2): [1, 2],
                (1, 3): [1, 2],
                (2, 4): [1, 2],
                (3, 5): [1, 2],
                (3, 6): [2, 3],
                (2, 7): [2, 3],
                (1, 8): [2, 3],
                (0, 9): [2, 3],
            }
            # for cells that don't overlap
            zone_defs = [
                # zone 1
                {
                    0: range(2),
                    1: range(3),
                    2: range(4),
                    3: range(5),
                    **{row: range(6) for row in range(4, 15)},
                },
                # zone 2
                {0: range(3, 9), 1: range(4, 8), 2: range(5, 7)},
                # zone 3
                {
                    1: range(9, 10),
                    2: range(8, 10),
                    3: range(7, 10),
                    **{row: range(6, 10) for row in range(4, 15)},
                },
            ]
            for row in range(15):
                for col in range(10):
                    if (row, col) in overlap and zone in overlap[(row, col)]:
                        # two zone overlap here
                        yield ((0, row, col), f"rch_{zone}", 0.5, None)
                    elif row in zone_defs[zone - 1] and col in zone_defs[zone - 1][row]:
                        # this cell belongs to one zone
                        yield ((0, row, col), f"rch_{zone}", 1.0, None)

        try:
            # load timeseries data
            with as_file(files(data).joinpath("data_rch_ts.csv")) as csv:
                rch_ts_df = pd.read_csv(csv)

            # generate three different recharge zones
            for zone in [1, 2, 3]:
                zone_data = list(_make_rch_stress_data(zone=zone))
                rch = fp.mf6.ModflowGwfrch(
                    model=getattr(self, "_model"),
                    fixed_cell=True,
                    auxiliary=["auxiliary", "multiplier"],
                    auxmultname="multiplier",
                    boundnames=True,
                    print_input=True,
                    print_flows=True,
                    save_flows=True,
                    maxbound=len(zone_data),
                    # list of (cellid, recharge, multiplier, boundname)
                    stress_period_data=zone_data,
                    filename=f"{self.sim_name}.rch{zone}",
                    pname=f"rch-zone_{zone}",
                )

                # add timeseries data to RCH zone
                fp.mf6.ModflowUtlts(
                    parent_package=rch,
                    time_series_namerecord=rch_ts_df.columns[zone],
                    interpolation_methodrecord="stepwise",
                    timeseries=rch_ts_df[
                        ["ts_time", rch_ts_df.columns[zone]]
                    ].to_records(index=False),
                    filename=f"{self.sim_name}.rch{zone}.ts",
                    pname=f"rch_{zone}_ts",
                )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_evt(self):
        """Add Evapotranspiration (EVT) package to model."""

        def _make_evt_stress_data() -> (
            Iterator[
                tuple[
                    tuple[int, int, int],
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    None,
                ]
            ]
        ):
            """Generate stress period data for evapotranspiration boundary condition.

            Yields
            ------
            ((layer, row, column), surface, rate, depth, pxdp_0, pxdp_1, petm_0, petm_1, petm0)
            """
            for row in range(15):
                for col in range(10):
                    yield ((0, row, col), 50.0, 0.0004, 10.0, 0.2, 0.5, 0.3, 0.1, None)

        try:
            fp.mf6.ModflowGwfevt(
                model=getattr(self, "_model"),
                maxbound=150,
                # number of ET segments
                # NSEG > 1: the PXDP and PETM arrays must be of size NSEG - 1 and be listed
                #   in order from the uppermost segment down
                nseg=3,
                # list of (cellid, surface, rate, depth, pxdp_0, pxdp_1, petm_0, petm_1, petm0)
                # surface: elevation of the ET surface (L)
                # rate: the maximum ET flux rate (L / T)
                # depth: ET extinction depth (L)
                # pxdp: the proportion of the ET extinction depth at the bottom of a segment
                # petm: the proportion of the maximum ET flux rate at the bottom of a segment
                # petm0: the proportion of the maximum ET flux rate that will apply when
                #   head is at or above the ET surface
                stress_period_data=list(_make_evt_stress_data()),
                filename=f"{self.sim_name}.evt",
                pname="evt",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_oc(self):
        """Add Output Control (OC) to model."""
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

    def _add_model_obs(self):
        """Add Observation (OBS) to model."""
        try:
            # for more observation types: see mf6io.pdf, pgs. 255-263
            fp.mf6.ModflowUtlobs(
                parent_model_or_package=getattr(self, "_model"),
                continuous={
                    f"{self.sim_name}.obs.head.csv": [
                        ("h3_13_8", "head", (2, 12, 7), None)
                    ],
                    f"{self.sim_name}.obs.flow.csv": [
                        ("icf1", "flow-ja-face", (0, 4, 5), (0, 5, 5))
                    ],
                },
                filename=f"{self.sim_name}.obs",
                pname="obs",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass
