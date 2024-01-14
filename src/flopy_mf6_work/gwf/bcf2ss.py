"""bcf2ss module"""
# pylint: disable=R0801
from os import PathLike
from typing import Iterator, Union

import flopy as fp
import matplotlib.pyplot as plt
import numpy as np


class BCF2SS:
    """BCF2SS example
    (see mf6examples.pdf, pages 2-1 through 2-5)

    Parameters
    ----------
    exe_name : str or PathLike
    sim_ws : str or PathLike
    newton : bool
        If True, generate the Newton-Raphson version of this example.
        Otherwise, generate the standard version of this example.

    Attributes
    ----------
    sim_name : str
        Simulation name is `gwf_bcf2ss`.
    is_newton : bool
        Indicates if the Newton-Raphson version of this example was generated.
    simulation : flopy.mf6.MFSimulation
    model : flopy.mf6.ModflowGwf
    """

    sim_name = "gwf_bcf2ss"

    def __init__(
        self, exe_name: Union[str, PathLike], sim_ws: Union[str, PathLike], newton: bool
    ):
        self._newton = newton

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
            save_flows=True,
            newtonoptions="NEWTON" if self._newton else None,
        )
        self._add_model_dis()
        self._add_model_npf()
        self._add_model_ic()
        self._add_model_riv()
        self._add_model_wel()
        self._add_model_rcha()
        self._add_model_oc()

    @property
    def is_newton(self):
        """Get the `newton` flag that was passed to the constructor."""
        return self._newton

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

    def plot_mapview_lower_kper0(self):
        """Generate a map-view plot of the lower aquifer during the first stress period
        with the river boundary condition.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, axes = plt.subplots()
        axes.set_title(
            "Map View of Lower Aquifer (1st stress period)", fontweight="bold"
        )
        mapview = fp.plot.PlotMapView(model=self._model, ax=axes, layer=1)
        mapview.plot_ibound()
        mapview.plot_grid()

        # plot stream
        mapview.plot_bc("RIV")

        # return completed figure to caller
        return fig

    def plot_mapview_lower_kper1(self):
        """Generate a map-view plot of the lower aquifer during the second stress period
        with well and river boundary conditions.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, axes = plt.subplots()
        axes.set_title(
            "Map View of Lower Aquifer (2nd stress period)", fontweight="bold"
        )
        mapview = fp.plot.PlotMapView(model=self._model, ax=axes, layer=1)
        mapview.plot_ibound()
        mapview.plot_grid()

        # plot wells
        mapview.plot_bc("WEL", kper=1)
        # plot stream
        mapview.plot_bc("RIV", kper=1)

        # return completed figure to caller
        return fig

    def _add_sim_tdis(self):
        """Add Temporal Discretization (TDIS) package to simulation.

        Two steady-state solutions were obtained to simulate natural conditions
        and pumping conditions.
        """
        try:
            fp.mf6.ModflowTdis(
                simulation=getattr(self, "_simulation"),
                time_units="days",
                nper=2,
                # list of (perlen, nstp, tsmult)
                # perlen (double): length of a stress period
                # nstp (int): number of time steps in a stress period
                # tsmult (double): multiplier for the length of successive time steps
                perioddata=[
                    # first stress period
                    (1.0, 1, 1.0),
                    # second stress period
                    (1.0, 1, 1.0),
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
                # dependent-variable change criterion for convergence of
                # the outer (nonlinear) iterations
                outer_dvclose=1e-6,
                # maximum number of outer (nonlinear) iterations
                outer_maximum=500,
                # maximum number of inner (linear) iterations
                inner_maximum=100,
                # dependent-variable change criterion for convergance of
                # the inner (linear) iterations
                inner_dvclose=1e-6,
                # rcloserecord = [inner_rclose, rclose_option]
                # inner_rclose: flow residual tolerance for convergence of
                #   the IMS linear solver
                # rclose_option: specific flow residual criterion used
                rcloserecord=[0.001, "strict"],
                # cg = preconditioned conjugate gradient method
                # bicgstab = preconditioned bi-conjugate gradient stabilized method
                linear_acceleration="bicgstab" if getattr(self, "_newton") else "cg",
                # used by the incomplete LU factorization preconditioners
                relaxation_factor=0.97,
                filename=f"{self.sim_name}.ims",
                pname="ims",
            )
        except AttributeError:
            # something doesn't exist, do nothing
            pass

    def _add_model_dis(self):
        """Add Structured Discretization (DIS) to model.

        The model consists of two layers - one for each aquifer.
        A uniform horizontal grid of 10 rows and 15 columns is used.

        Because horizontal flow in the confining bed is small compared to
        horizontal flow in the aquifers and storage is not a factor in
        steady-state simulations, the confining bed is not treated as a
        separate layer.
        """
        try:
            fp.mf6.ModflowGwfdis(
                model=getattr(self, "_model"),
                length_units="feet",
                nlay=2,
                nrow=10,
                ncol=15,
                # column spacing in the row direction
                delr=500.0,
                # row spacing in the column direction
                delc=500.0,
                # top elevation for each cell in the top model layer
                top=150.0,
                botm=[
                    # bottom of upper aquifer
                    50.0,
                    # bottom of lower aquifer
                    -50.0,
                ],
                filename=f"{self.sim_name}.dis",
                pname="dis",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_npf(self):
        """Add Node Property Flow (NPF) package to model.

        A horizontal hydraulic conductivity of 10 and 5 ft/day is specified
        for the upper and lower aquifers, respectively; the horizontal conductivity
        of the lower aquifer is calculated based on the transmissivity and layer thickness
        used in the original problem. The vertical hydraulic conductivity of the
        confining units is calculated from the vertical conductance of the confining beds
        defined in the original problem.
        """
        try:
            lay1_wetdry = np.ones((10, 15))
            lay1_wetdry[(2, 7), 3] = -1
            lay1_wetdry[:, 8:] = -1
            lay1_wetdry *= 2

            fp.mf6.ModflowGwfnpf(
                model=getattr(self, "_model"),
                rewet_record=[
                    # factor included in the calculation of head that is initially
                    # established at a cell when that cell is converted from dry to wet
                    "wetfct",
                    1.0,
                    # iteration interval for attempting to wet cells
                    "iwetit",
                    1,
                    # flag that determines which equation is used to define the initial head
                    # at cells that become wet
                    "ihdwet",
                    0,
                ]
                if not getattr(self, "_newton")
                else None,
                save_specific_discharge=True,
                icelltype=[
                    # upper aquifer is unconfined/convertible
                    1,
                    # lower aquifer is confined
                    0,
                ],
                # (horizontal) hydraulic conductivity
                k=[
                    # upper aquifer
                    10.0,
                    # lower aquifer
                    5.0,
                ],
                # (vertical) hydraulic conductivity
                k33=0.1,
                # a combination of the wetting threshold and a flag to indicate which
                #   neighboring cells can cause a cell to become wet
                # wetdry < 0: only a cell below a dry cell can cause the cell to become wet
                # wetdry > 0: the cell below a dry cell and horizontally adjacent cells
                #   can cause a cell to become wet
                wetdry=[
                    # upper aquifer
                    lay1_wetdry,
                    # lower aquifer
                    0.0,
                ]
                if not getattr(self, "_newton")
                else None,
                filename=f"{self.sim_name}.npf",
                pname="npf",
            )
        except AttributeError:
            # something doesn't exist, do nothing
            pass

    def _add_model_ic(self):
        """Add Initial Conditions (IC) package to model.

        An initial head of zero ft is specified in all model layers.
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

    def _add_model_riv(self):
        """Add River (RIV) package to model.

        Flow out of the model is partly from a stream represented by river (RIV)
        package cells in the lower aquifer. River cells are located in column 15
        of every row in the lower aquifer.
        """

        def _make_riv_iter() -> (
            Iterator[tuple[tuple[int, int, int], float, float, float]]
        ):
            """Generate river data for model.

            Yields
            ------
            ((layer, row, column), stage, cond, rbot)
            """
            for row in range(10):
                yield ((1, row, 14), 0.0, 10000.0, -5.0)

        try:
            fp.mf6.ModflowGwfriv(
                model=getattr(self, "_model"),
                # maximum number of river cells that will be specified
                # for use during any stress period
                maxbound=10,
                # list of (cellid, stage, cond, rbot)
                # cellid: (layer, row, column)
                # stage: head in the river
                # cond: riverbed hydraulic conductance
                # rbot: elevation of the bottom of the riverbed
                stress_period_data=list(_make_riv_iter()),
                filename=f"{self.sim_name}.riv",
                pname="riv_0",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_wel(self):
        """Add Well (WEL) package to model.

        Flow out of the model is partly from discharging wells represented by
        well (WEL) package cells in the lower aquifer. Two wells are included
        in the second stress period.
        """
        try:
            fp.mf6.ModflowGwfwel(
                model=getattr(self, "_model"),
                maxbound=2,
                stress_period_data={
                    # wells active in second stress period
                    # both wells in lower aquifer
                    1: [((1, 2, 3), -35000.0), ((1, 7, 3), -35000.0)]
                },
                filename=f"{self.sim_name}.wel",
                pname="wel_0",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass

    def _add_model_rcha(self):
        """Add Recharge (RCH) package to model.

        Flow into the system is from infiltration from precipitation and was
        represented using the recharge (RCH) package. A constant recharge rate
        was specified for every cell in the upper aquifer.
        """
        try:
            fp.mf6.ModflowGwfrcha(
                model=getattr(self, "_model"),
                # recharge fllux rate (L / T)
                recharge=0.004,
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
                filename=f"{self.sim_name}.oc",
                pname="oc",
            )
        except AttributeError:
            # model doesn't exist, do nothing
            pass
