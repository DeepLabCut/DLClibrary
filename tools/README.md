# Developer tools useful for maintaining the DeepLabCut repository

## Code headers

The code headers can be standardized by running

``` bash
python tools/update_license_headers.py
```

from the repository root (this needs `pip install licenseheaders`)

You can edit the `NOTICE.yml` to update the header.


## Running tests [with more details](https://docs.pytest.org/en/latest/how-to/capture-stdout-stderr.html)

``` bash
pytest -s
```
