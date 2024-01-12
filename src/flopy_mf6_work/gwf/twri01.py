"""twri01 module"""
from os import PathLike
from typing import Iterator, Union

import flopy as fp
import matplotlib.pyplot as plt


class TWRI:
    """TWRI example
    (see mf6examples.pdf, pages 1-1 through 1-3)

    Parameters
    ----------
    exe_name : str or PathLike
    sim_ws : str or PathLike

    Attributes
    ----------
    sim_name : str
        Simulation name is `gwf_twri01`.
    simulation : flopy.mf6.MFSimulation
    model : flopy.mf6.ModflowGwf
    """

    sim_name = "gwf_twri01"

    def __init__(
        self,
        exe_name: Union[str, PathLike],
        sim_ws: Union[str, PathLike],
    ):
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
        )
        self._add_model_dis()
        self._add_model_npf()
        self._add_model_ic()
        self._add_model_chd()
        self._add_model_drn()
        self._add_model_wel()
        self._add_model_rcha()
        self._add_model_oc()

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

    def plot_mapview_unconfined(self):
        """Generate a map-view plot of the unconfined aquifer in the first layer
        with constant-head, drain, and well boundary conditions.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, ax = plt.subplots()
        ax.set_title("Map View of Unconfined Aquifer Layer", fontweight="bold")
        mapview = fp.plot.PlotMapView(model=self._model, ax=ax)
        mapview.plot_ibound()
        mapview.plot_grid()

        # plot constant-head cells
        mapview.plot_bc("CHD")
        # plot drain cells
        mapview.plot_bc("DRN")
        # plot wells
        mapview.plot_bc("WEL")

        # return completed figure to caller
        return fig

    def plot_mapview_middle_confined(self):
        """Generate a map-view plot of the middle confined aquifer in the third layer
        with constant-head and well boundary conditions.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, ax = plt.subplots()
        ax.set_title("Map View of Middle Confined Aquifer Layer", fontweight="bold")
        mapview = fp.plot.PlotMapView(model=self._model, ax=ax, layer=2)
        mapview.plot_ibound()
        mapview.plot_grid()

        # plot constant-head cells
        mapview.plot_bc("CHD")
        # plot wells
        mapview.plot_bc("WEL")

        # return completed figure to caller
        return fig

    def plot_mapview_lower_confined(self):
        """Generate a map-view plot of the lower confined aquifer in the fifth layer
        with well boundary conditions.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, ax = plt.subplots()
        ax.set_title("Map View of Lower Confined Aquifer Layer", fontweight="bold")
        mapview = fp.plot.PlotMapView(model=self._model, ax=ax, layer=4)
        mapview.plot_ibound()
        mapview.plot_grid()

        # plot wells
        mapview.plot_bc("WEL")

        # return completed figure to caller
        return fig

    def _add_sim_tdis(self):
        """Add Temporal Discretization (TDIS) packge to simulation."""
        try:
            fp.mf6.ModflowTdis(
                simulation=getattr(self, "_simulation"),
                time_units="seconds",
                nper=1,
                perioddata=[(86400.0, 1, 1.0)],
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
                rcloserecord=[(1e-6, "strict")],
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
                length_units="FEET",
                nlay=5,
                nrow=15,
                ncol=15,
                delr=5000.0,
                delc=5000.0,
                top=200.0,
                botm=[-150.0, -200.0, -300.0, -350.0, -450.0],
                filename=f"{self.sim_name}.dis",
                pname="dis",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_npf(self):
        """Add Node Property Flow (NPF) package to model."""
        try:
            k = [0.001, 1e-8, 0.0001, 5e-7, 0.0002]
            fp.mf6.ModflowGwfnpf(
                model=getattr(self, "_model"),
                cvoptions=[(True, "DEWATERED")],
                perched=True,
                save_specific_discharge=True,
                icelltype=[1, 0, 0, 0, 0],
                k=k,
                k33=k,
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
                strt=0.0,
                filename=f"{self.sim_name}.ic",
                pname="ic",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_chd(self):
        """Add Constant-Head (CHD) package to model."""

        def _make_chd_iter() -> Iterator[tuple[tuple[int, int, int], float]]:
            """Generate constant-head data for model.
            The first column of the first and third layers should be set to 0.0 ft.

            Yields
            ------
            ((layer, row, column), head)
            """
            for layer in [0, 2]:
                for row in range(15):
                    yield ((layer, row, 0), 0.0)

        try:
            fp.mf6.ModflowGwfchd(
                model=getattr(self, "_model"),
                maxbound=30,
                stress_period_data=list(_make_chd_iter()),
                filename=f"{self.sim_name}.chd",
                pname="chd_0",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_drn(self):
        """Add Drain (DRN) package to model."""

        def _make_drn_iter() -> Iterator[tuple[tuple[int, int, int], float, float]]:
            """Generate drain data for model.

            Yields
            ------
            ((layer, row, column), elev, cond)
            """
            elev = [0, 0, 10, 20, 30, 50, 70, 90, 100]
            for col in range(1, 10):
                yield ((0, 7, col), float(elev[col - 1]), 1.0)

        try:
            fp.mf6.ModflowGwfdrn(
                model=getattr(self, "_model"),
                maxbound=9,
                stress_period_data=list(_make_drn_iter()),
                filename=f"{self.sim_name}.drn",
                pname="drn_0",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_wel(self):
        """Add Well (WEL) package to model."""

        def _make_wel_iter() -> Iterator[tuple[tuple[int, int, int], float]]:
            """Generate well data for model.

            Yields
            ------
            ((layer, row, column), q)
            """
            # wells in unconfined aquifer
            for row in range(8, 13, 2):
                for col in range(7, 14, 2):
                    yield ((0, row, col), -5.0)

            # wells in middle confined aquifer
            yield ((2, 3, 5), -5.0)
            yield ((2, 5, 11), -5.0)

            # well in lower confined aquifer
            yield ((4, 4, 10), -5.0)

        try:
            fp.mf6.ModflowGwfwel(
                model=getattr(self, "_model"),
                maxbound=15,
                stress_period_data=list(_make_wel_iter()),
                filename=f"{self.sim_name}.wel",
                pname="wel_0",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_rcha(self):
        """Add Recharge (RCH) package to model (array-based version)."""
        try:
            fp.mf6.ModflowGwfrcha(
                model=getattr(self, "_model"),
                recharge=3e-8,
                filename=f"{self.sim_name}.rcha",
                pname="rcha_0",
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
                # printrecord=None,
                filename=f"{self.sim_name}.oc",
                pname="oc",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass
