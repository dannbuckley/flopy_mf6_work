"""fhb module"""
# pylint: disable=R0801
from importlib.resources import files
from os import PathLike
from typing import Iterator, Union

import flopy as fp
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd

from . import data


class FHB:
    """FHB example
    (see mf6examples.pdf, pages 3-1 through 3-4)

    Parameters
    ----------
    exe_name : str or PathLike
    sim_ws : str or PathLike

    Attributes
    ----------
    sim_name : str
        Simulation name is `gwf_fhb`.
    """

    sim_name = "gwf_fhb"

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
        self._add_model_chd()
        self._add_model_wel()
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

    def plot_obs_head(self) -> Figure | None:
        """Generate a plot of the observed heads from the completed simulation.

        Returns
        -------
        fig : matplotlib.figure.Figure
            If the head observations exist.
        """
        try:
            # load head data
            head_obs_df = pd.read_csv(
                self._simulation.sim_path / f"{self.sim_name}.obs.head.csv"
            )
            fig, axes = plt.subplots()
            axes.plot("time", "H1_2_1", "o-", data=head_obs_df, label="H1_2_1")
            axes.plot("time", "H1_2_10", "o-", data=head_obs_df, label="H1_2_10")
            axes.legend()

            axes.set_title("Observed Heads", fontweight="bold")
            axes.set_xlabel("Time (d)")
            axes.set_ylabel("Head (m)")

            # return completed figure to caller
            return fig
        except FileNotFoundError:
            # head observations don't exist
            return None

    def plot_obs_flow(self) -> Figure | None:
        """Generate a plot of the observed flows from the completed simulation.

        Returns
        -------
        fig : matplotlib.figure.Figure
            If the flow observations exist.
        """
        try:
            # load flow data
            flow_obs_df = pd.read_csv(
                self._simulation.sim_path / f"{self.sim_name}.obs.flow.csv"
            )
            fig, axes = plt.subplots()
            axes.plot("time", "ICF1", "o-", data=flow_obs_df, label="ICF1")
            axes.legend()

            axes.set_title(
                "Observed Flow from Cell (1, 2, 1) to Cell (1, 2, 2)", fontweight="bold"
            )
            axes.set_xlabel("Time (d)")
            axes.set_ylabel("Flow (m$^3$/d)")

            # return completed figure to caller
            return fig
        except FileNotFoundError:
            # flow observations don't exist
            return None

    def _add_sim_tdis(self):
        """Add Temporal Discretization (TDIS) package to simulation."""
        try:
            fp.mf6.ModflowTdis(
                simulation=getattr(self, "_simulation"),
                time_units="days",
                nper=3,
                perioddata=[
                    # first stress period, transient
                    (400.0, 10, 1.0),
                    # second stress period, transient
                    (200.0, 4, 1.0),
                    # third stress period, transient
                    (400.0, 6, 1.0),
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
                nlay=1,
                nrow=3,
                ncol=10,
                delr=1000.0,
                delc=1000.0,
                top=50.0,
                botm=-200.0,
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
                save_specific_discharge=True,
                # layer is confined
                icelltype=0,
                k=20.0,
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
                # starting head
                strt=0.0,
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
                storagecoefficient=True,
                # confined storage is used
                iconvert=0,
                # storage coefficient (since flag is set)
                ss=1e-6,
                transient={0: True},
                filename=f"{self.sim_name}.sto",
                pname="sto",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_chd(self):
        """Add Constant-Head (CHD) package to model."""

        def _make_chd_stress_data() -> Iterator[tuple[tuple[int, int, int], str]]:
            """Generate constant-head stress period data."""
            for row in range(3):
                yield ((0, row, 9), "chdhead")

        try:
            chd = fp.mf6.ModflowGwfchd(
                model=getattr(self, "_model"),
                maxbound=3,
                stress_period_data=list(_make_chd_stress_data()),
                filename=f"{self.sim_name}.chd",
                pname="chd",
            )

            # load timeseries data
            chd_ts_df = pd.read_csv(files(data) / "data_chd_ts.csv")
            # add timeseries data to CHD package
            fp.mf6.ModflowUtlts(
                parent_package=chd,
                time_series_namerecord=chd_ts_df.columns[1],
                interpolation_methodrecord="linearend",
                timeseries=chd_ts_df.to_records(index=False),
                filename=f"{self.sim_name}.chd.ts",
                pname="chd_ts",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_wel(self):
        """Add Well (WEL) package to model."""
        try:
            wel = fp.mf6.ModflowGwfwel(
                model=getattr(self, "_model"),
                maxbound=1,
                stress_period_data=[((0, 1, 0), "flowrate")],
                filename=f"{self.sim_name}.wel",
                pname="wel",
            )

            # load timeseries data
            wel_ts_df = pd.read_csv(files(data) / "data_wel_ts.csv")
            # add timeseries data to WEL package
            fp.mf6.ModflowUtlts(
                parent_package=wel,
                time_series_namerecord=wel_ts_df.columns[1],
                interpolation_methodrecord="linearend",
                timeseries=wel_ts_df.to_records(index=False),
                filename=f"{self.sim_name}.wel.ts",
                pname="wel_ts",
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
            fp.mf6.ModflowUtlobs(
                parent_model_or_package=getattr(self, "_model"),
                continuous={
                    f"{self.sim_name}.obs.flow.csv": [
                        ("icf1", "flow-ja-face", (0, 1, 1), (0, 1, 0))
                    ],
                    f"{self.sim_name}.obs.head.csv": [
                        ("h1_2_1", "head", (0, 1, 0), None),
                        ("h1_2_10", "head", (0, 1, 9), None),
                    ],
                },
                filename=f"{self.sim_name}.obs",
                pname="obs",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass
