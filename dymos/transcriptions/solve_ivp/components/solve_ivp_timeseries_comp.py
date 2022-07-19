from numpy.polynomial import Polynomial

from dymos.transcriptions.common.timeseries_output_comp import TimeseriesOutputCompBase


class SolveIVPTimeseriesOutputComp(TimeseriesOutputCompBase):
    """
    Class definition for SolveIVPTimeseriesOutputComp.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def initialize(self):
        """
        Declare component options.
        """
        super(SolveIVPTimeseriesOutputComp, self).initialize()

        self.options.declare('output_nodes_per_seg', default=None, types=(int,), allow_none=True,
                             desc='If None, results are provided at the all nodes within each'
                                  'segment.  If an int (n) then results are provided at n '
                                  'equally distributed points in time within each segment.')

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        grid_data = self.options['input_grid_data']
        if self.options['output_nodes_per_seg'] is None:
            self.num_nodes = grid_data.num_nodes
        else:
            self.num_nodes = grid_data.num_segments * self.options['output_nodes_per_seg']

    def _add_output_configure(self, name, units, shape, desc, rate_src=None):
        """
        Add a single timeseries output.

        Can be called by parent groups in configure.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array.
            Default is None.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.
        desc : str
            description of the timeseries output variable.
        rate_src : str or None
            If not None, timeseries output is a rate and rate_src is the original variable name.
        """
        self._has_rate |= rate_src is not None

        varshape = (self.num_nodes,) + shape
        input_name = f'all_values:{name}'

        self.add_input(input_name, shape=varshape,  units=units, desc=desc)
        self.add_output(name, shape=varshape, units=units, desc=desc)

        self._vars[name] = (input_name, name, shape, rate_src)

    def compute(self, inputs, outputs):
        """
        Compute component outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        if self._has_rate:
            nodes_per_seg = self.options['output_nodes_per_seg']
            nsegs = self.num_nodes // nodes_per_seg
            time = inputs['all_values:time'][:, 0]
            for iname, oname, _, rate_src in self._vars.values():
                if rate_src is None:
                    outputs[oname] = inputs[iname]
                else:
                    ins = inputs[rate_src][:, 0]
                    outs = outputs[oname][:, 0]
                    start = end = 0
                    for i in range(nsegs):
                        end += nodes_per_seg
                        poly = Polynomial.fit(time[start:end], ins[start:end], end - start - 1)
                        deriv = poly.deriv(1)
                        outs[start:end] = deriv(time[start:end])
                        start = end
        else:
            outputs.set_val(inputs.asarray())
