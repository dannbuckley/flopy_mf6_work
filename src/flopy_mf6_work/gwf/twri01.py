"""twri01 module"""
# pylint: disable=R0801
from os import PathLike
from pathlib import Path
from typing import Iterator, Union

import flopy as fp
from matplotlib.figure import Figure
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

        # setup groundwater flow model
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

    def plot_mapview_unconfined(self) -> Figure:
        """Generate a map-view plot of the unconfined aquifer in the first layer
        with constant-head, drain, and well boundary conditions.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, axes = plt.subplots()
        axes.set_title("Map View of Unconfined Aquifer Layer", fontweight="bold")
        mapview = fp.plot.PlotMapView(model=self._model, ax=axes)
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

    def plot_heads_unconfined(self) -> Figure | None:
        """Generate a map-view plot of the unconfined aquifer in the first layer
        with heads.

        Returns
        -------
        fig : matplotlib.figure.Figure
            If the heads output file exists.
        """
        try:
            # load head file
            hds = fp.utils.binaryfile.HeadFile(
                Path(self._simulation.sim_path, f"{self.sim_name}.hds")
            )
            head_data = hds.get_data()

            fig, axes = plt.subplots()
            axes.set_title("Unconfined Aquifer w/ Heads", fontweight="bold")
            mapview = fp.plot.PlotMapView(model=self._model, ax=axes, layer=0)
            mapview.plot_ibound()
            mapview.plot_grid()

            # filled head contour
            hds_fill = mapview.plot_array(head_data, cmap="rainbow_r", alpha=0.4)
            plt.colorbar(hds_fill)

            # head contour lines
            hds_line = mapview.contour_array(head_data, colors="black")
            plt.clabel(hds_line, fmt="%.0f")

            # plot drain tubes
            mapview.plot_bc("DRN")
            # plot wells
            mapview.plot_bc("WEL")

            # return completed figure to caller
            return fig
        except FileNotFoundError:
            # heads file doesn't exist
            return None

    def plot_mapview_middle_confined(self) -> Figure:
        """Generate a map-view plot of the middle confined aquifer in the third layer
        with constant-head and well boundary conditions.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, axes = plt.subplots()
        axes.set_title("Map View of Middle Confined Aquifer Layer", fontweight="bold")
        mapview = fp.plot.PlotMapView(model=self._model, ax=axes, layer=2)
        mapview.plot_ibound()
        mapview.plot_grid()

        # plot constant-head cells
        mapview.plot_bc("CHD")
        # plot wells
        mapview.plot_bc("WEL")

        # return completed figure to caller
        return fig

    def plot_heads_middle_confined(self) -> Figure | None:
        """Generate a map-view plot of the middle confined aquifer in the third layer
        with heads.

        Returns
        -------
        fig : matplotlib.figure.Figure
            If the heads output file exists.
        """
        try:
            # load head file
            hds = fp.utils.binaryfile.HeadFile(
                Path(self._simulation.sim_path, f"{self.sim_name}.hds")
            )
            head_data = hds.get_data()

            fig, axes = plt.subplots()
            axes.set_title("Middle Confined Aquifer w/ Heads", fontweight="bold")
            mapview = fp.plot.PlotMapView(model=self._model, ax=axes, layer=2)
            mapview.plot_ibound()
            mapview.plot_grid()

            # filled head contour
            hds_fill = mapview.plot_array(head_data, cmap="rainbow_r", alpha=0.4)
            plt.colorbar(hds_fill)

            # head contour lines
            hds_line = mapview.contour_array(head_data, colors="black")
            plt.clabel(hds_line, fmt="%.0f")

            # plot wells
            mapview.plot_bc("WEL")

            # return completed figure to caller
            return fig
        except FileNotFoundError:
            # heads file doesn't exist
            return None

    def plot_mapview_lower_confined(self) -> Figure:
        """Generate a map-view plot of the lower confined aquifer in the fifth layer
        with well boundary conditions.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, axes = plt.subplots()
        axes.set_title("Map View of Lower Confined Aquifer Layer", fontweight="bold")
        mapview = fp.plot.PlotMapView(model=self._model, ax=axes, layer=4)
        mapview.plot_ibound()
        mapview.plot_grid()

        # plot wells
        mapview.plot_bc("WEL")

        # return completed figure to caller
        return fig

    def plot_heads_lower_confined(self) -> Figure | None:
        """Generate a map-view plot of the lower confined aquifer in the fifth layer
        with heads.

        Returns
        -------
        fig : matplotlib.figure.Figure
            If the heads output file exists.
        """
        try:
            # load head file
            hds = fp.utils.binaryfile.HeadFile(
                Path(self._simulation.sim_path, f"{self.sim_name}.hds")
            )
            head_data = hds.get_data()

            fig, axes = plt.subplots()
            axes.set_title("Lower Confined Aquifer w/ Heads", fontweight="bold")
            mapview = fp.plot.PlotMapView(model=self._model, ax=axes, layer=4)
            mapview.plot_ibound()
            mapview.plot_grid()

            # filled head contour
            hds_fill = mapview.plot_array(head_data, cmap="rainbow_r", alpha=0.4)
            plt.colorbar(hds_fill)

            # head contour lines
            hds_line = mapview.contour_array(head_data, colors="black")
            plt.clabel(hds_line, fmt="%.0f")

            # plot well
            mapview.plot_bc("WEL")

            # return completed figure to caller
            return fig
        except FileNotFoundError:
            # heads file doesn't exist
            return None

    def _add_sim_tdis(self):
        """Add Temporal Discretization (TDIS) packge to simulation.

        A single steady-stress period with a total length of 86,400 seconds
        (1 day) is simulated.
        """
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
        """Add Structured Discretization (DIS) to model.

        There are three simulated aquifers, which are separated from each other by confining layers.
        The confining beds are 50 ft thick and are explicitly simulated as
        model layers 2 and 4, respectively.
        Each layer is a square 75,000 ft on a side and is divided into a
        grid with 15 rows and 15 columns,
        which forms squares 5,000 ft on a side.
        """
        try:
            fp.mf6.ModflowGwfdis(
                model=getattr(self, "_model"),
                length_units="FEET",
                nlay=5,
                nrow=15,
                ncol=15,
                delr=5000.0,
                delc=5000.0,
                # top of unconfined aquifer
                top=200.0,
                botm=[
                    # bottom of unconfined aquifer
                    -150.0,
                    # bottom of first confining unit
                    -200.0,
                    # bottom of middle confined aquifer
                    -300.0,
                    # bottom of second confining unit
                    -350.0,
                    # bottom of lower confined aquifer
                    -450.0,
                ],
                filename=f"{self.sim_name}.dis",
                pname="dis",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_npf(self):
        """Add Node Property Flow (NPF) package to model.

        The transmissivity of the middle and lower aquifers was converted to
        a horizontal hydraulic conductivity using the layer thickness. The
        vertical hydraulic conductivity in the aquifers was set equal to the
        horizontal hydraulic conductivity. The vertical hydraulic conductivity
        of the confining units was calculated from the vertical conductance
        of the confining beds defined in the original problem and the
        confining unit thickness; the horizontal hydraulic conductivity of
        the confining bed was set to the vertical hydraulic conductivity
        and results in vertical flow in the confining unit.
        """
        try:
            k = [
                # unconfined aquifer
                0.001,
                # first confining unit
                1e-8,
                # middle confined aquifer
                0.0001,
                # second confining unit
                5e-7,
                # lower confined aquifer
                0.0002,
            ]
            fp.mf6.ModflowGwfnpf(
                model=getattr(self, "_model"),
                cvoptions=[(True, "DEWATERED")],
                # when a cell is overlying a dewatered convertible cell, the head difference
                # used in Darcy's Law is equal to the head in the overlying cell minus
                # the bottom elevation of the overlying cell
                perched=True,
                save_specific_discharge=True,
                icelltype=[
                    # first layer: saturated thickness varies with computed head when
                    # head is below the cell top
                    # (i.e., first layer is unconfined)
                    1,
                    # remaining layers: saturated thickness is held constant
                    0,
                    0,
                    0,
                    0,
                ],
                k=k,
                k33=k,
                filename=f"{self.sim_name}.npf",
                pname="npf",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_ic(self):
        """Add Initial Conditions (IC) package to model.

        An initial head of zero ft was specified in all model layers.
        Any initial head exceeding the bottom of model layer 1 (-150 ft)
        could be specified since the model is steady-state.
        """
        try:
            fp.mf6.ModflowGwfic(
                model=getattr(self, "_model"),
                # initial (starting) head
                strt=0.0,
                filename=f"{self.sim_name}.ic",
                pname="ic",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_chd(self):
        """Add Constant-Head (CHD) package to model.

        Flow out of the model is partly from a lake represented by
        constant head (CHD) package cells in the unconfined and middle aquifers.
        """

        def _make_chd_iter() -> Iterator[tuple[tuple[int, int, int], float]]:
            """Generate constant-head data for model.
            The first column of the first and third layers should be set to 0.0 ft.

            Yields
            ------
            ((layer, row, column), head)
            """
            for layer in [0, 2]:
                # layer = 0: unconfined aquifer
                # layer = 2: middle confined aquifer
                for row in range(15):
                    # apply constant-head to entire first column
                    yield ((layer, row, 0), 0.0)

        try:
            fp.mf6.ModflowGwfchd(
                model=getattr(self, "_model"),
                # maximum number of constant-head cells that will be specified
                # for use during any stress period
                maxbound=30,
                # list of (cellid, head)
                # cellid: (layer, row, column)
                # head: 0
                stress_period_data=list(_make_chd_iter()),
                filename=f"{self.sim_name}.chd",
                pname="chd_0",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_drn(self):
        """Add Drain (DRN) package to model.

        Flow out of the model is partly from buried drain tubes represented by
        drain (DRN) pacakge cells in model layer 1.
        """

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
                # maximum number of drain cells that will be specified
                # for use during any stress period
                maxbound=9,
                # list of (cellid, elev, cond)
                # cellid: (layer, row, column)
                # elev: elevation of the drain
                # cond: hydraulic conductance of the interface between the aquifer
                #   and the drain
                stress_period_data=list(_make_drn_iter()),
                filename=f"{self.sim_name}.drn",
                pname="drn_0",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_wel(self):
        """Add Well (WEL) package to model.

        Flow out of the model is partly from discharging wells represented by
        well (WEL) package cells in all three aquifers.
        """

        def _make_wel_iter() -> Iterator[tuple[tuple[int, int, int], float]]:
            """Generate well data for model.

            Yields
            ------
            ((layer, row, column), q)
            """
            # wells in unconfined aquifer
            for row in range(8, 13, 2):
                for col in range(7, 14, 2):
                    # negative well rate indicates discharge (extraction)
                    yield ((0, row, col), -5.0)

            # wells in middle confined aquifer
            yield ((2, 3, 5), -5.0)
            yield ((2, 5, 11), -5.0)

            # well in lower confined aquifer
            yield ((4, 4, 10), -5.0)

        try:
            fp.mf6.ModflowGwfwel(
                model=getattr(self, "_model"),
                # maximum number of well cells that will be specified
                # for use during any stress period
                maxbound=15,
                # list of (cellid, q)
                # cellid: (layer, row, column)
                # q: volumetric well rate
                stress_period_data=list(_make_wel_iter()),
                filename=f"{self.sim_name}.wel",
                pname="wel_0",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_rcha(self):
        """Add Recharge (RCH) package to model (array-based version).

        Flow into the system is from infiltration from precipitation and was
        represented using the recharge (RCH) package. A constant recharge rate
        was specified for every cell in model layer 1.
        """
        try:
            fp.mf6.ModflowGwfrcha(
                model=getattr(self, "_model"),
                # recharge flux rate (L / T)
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
                # output file for budget info
                budget_filerecord=[f"{self.sim_name}.cbc"],
                # output file for head info
                head_filerecord=[f"{self.sim_name}.hds"],
                # list of (rtype, ocsetting)
                # rtype: type of info
                # ocsetting: which steps
                saverecord=[("head", "all"), ("budget", "all")],
                filename=f"{self.sim_name}.oc",
                pname="oc",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass
