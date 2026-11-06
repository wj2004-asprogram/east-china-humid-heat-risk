# Data sources, attribution, and licenses

The notices below apply to the identified data, independently of the
[code and original-content licensing status](LICENSE.md). Public availability
of a source does not by itself grant redistribution rights.

## ERA5 climate data and derived results

ECMWF Reanalysis data (ERA5), © European Centre for Medium-Range Weather
Forecasts (ECMWF), retrieved via the Copernicus Climate Data Store.
Licensed under [Creative Commons Attribution 4.0 International
(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Dataset: [ERA5 hourly data on single levels from 1940 to
present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview).
Dataset DOI: [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47).
Method reference: Hersbach et al. (2020), *The ERA5 global reanalysis*,
[doi:10.1002/qj.3803](https://doi.org/10.1002/qj.3803).

The published climate tables, heat-index statistics, features, and model
results are **modified products based on ERA5**, not unmodified ECMWF products.
Changes include temperature-unit conversion, relative-humidity and heat-index
calculations, Beijing-time daily aggregation, land masking, lagged features,
and statistical modelling. The project authors are responsible for those
transformations and interpretations. ECMWF and Copernicus do not endorse the
factory assessments or research conclusions.

Raw ERA5 NetCDF files are not distributed in this Git repository. When sharing
derived climate data or graphics, retain this attribution, the license link,
and identification of the transformations. See the
[ECMWF attribution guidance](https://confluence.ecmwf.int/spaces/DAC/pages/576396026/How%2Bcan%2BI%2Battribute%2Band%2Bcite%2BECMWF%2Bdata).

## OpenStreetMap location data

© OpenStreetMap contributors. The OSM-derived names, coordinates, factory
geometries, and related location-database content are made available under the
[Open Data Commons Open Database License 1.0
(ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).
See [OpenStreetMap copyright and attribution](https://www.openstreetmap.org/copyright).

This includes the relevant fields in `data/site_inputs/`, derived spatial
tables and site mappings, and their copies in the dashboards and reports.
Retain attribution and ODbL notices; distribution of an adapted database must
follow the ODbL requirements. This does not license the project's code under
ODbL. OSM Wiki documentation is separately licensed under
[CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/).

## Government and company evidence

Government notices, company websites, certificates, and environmental-review
documents support the recorded factual identifiers and source links. No open
redistribution license was established for their full text. Original HTML,
PDF, and other source snapshots are therefore retained in the local audit
archive and are not included in the public repository. The public release
contains factual fields, source URLs, retrieval dates, and hashes for
traceability, without claiming a license over the source documents.

Factory identity and location evidence is a public-source snapshot dated
2026-10-29. Names and coordinates do not establish current operation, on-site
verification, worker exposure, safety compliance, or financial loss. Unverified
candidate locations retain their candidate status.

## Publication provenance

The public release separates unchanged scientific artifacts from redacted
execution records and publication changes. Its publication manifest records
source and published hashes. Historical scientific seals document the original
local experiment; they do not extend third-party licenses or certify that a
redacted publication tree is byte-identical to that local archive.
