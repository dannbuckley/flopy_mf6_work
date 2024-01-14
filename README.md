# flopy_mf6_work
Learning MODFLOW 6 using the FloPy library

Going through the examples in the [MODFLOW 6 software download](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model):
- Models are in the `examples/` directory of the download
- Descriptions for the examples are in `doc/mf6examples.pdf` within the download

## Examples

<table>
  <thead>
    <tr>
      <th scope="col">MF6 Folder</th>
      <th scope="col">Description</th>
      <th scope="col"></th>
      <th scope="col"></th>
      <th scope="col"></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ex-gwf-twri01</td>
      <td>TWRI</td>
      <td><a href="https://github.com/dannbuckley/flopy_mf6_work/blob/main/notebooks/learning/gwf_twri01.ipynb">Learning</a></td>
      <td><a href="https://github.com/dannbuckley/flopy_mf6_work/blob/main/src/flopy_mf6_work/gwf/twri01.py">Implementation</a></td>
      <td><a href="https://github.com/dannbuckley/flopy_mf6_work/blob/main/notebooks/testing/gwf_twri01/gwf_twri01.ipynb">Testing</a></td>
    </tr>
    <tr>
      <td>ex-gwf-bcf2ss-p[01, 02]a</td>
      <td>BCF2SS</td>
      <td><a href="https://github.com/dannbuckley/flopy_mf6_work/blob/main/notebooks/learning/gwf_bcf2ss.ipynb">Learning</a></td>
      <td><a href="https://github.com/dannbuckley/flopy_mf6_work/blob/main/src/flopy_mf6_work/gwf/bcf2ss.py">Implementation</a></td>
      <td><a href="https://github.com/dannbuckley/flopy_mf6_work/blob/main/notebooks/testing/gwf_bcf2ss/gwf_bcf2ss.ipynb">Testing</a></td>
    </tr>
  </tbody>
</table>

## Development

Docstrings should follow the [numpydoc style guide](https://numpydoc.readthedocs.io/en/latest/format.html).
