BANDS_AND_SCL = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"],
      units: "DN"
    }],
    // Single response with 7 bands: B02,B03,B04,B08,B11,B12,SCL.
    output: { id: "default", bands: 7, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  return [
    sample.B02 / 10000,
    sample.B03 / 10000,
    sample.B04 / 10000,
    sample.B08 / 10000,
    sample.B11 / 10000,
    sample.B12 / 10000,
    sample.SCL
  ];
}
"""

BANDS_HARMONIZED_AND_SCL = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"],
      units: "DN"
    }],
    // Single response with 11 bands:
    // B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12,SCL
    output: { id: "default", bands: 11, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  return [
    sample.B02 / 10000,
    sample.B03 / 10000,
    sample.B04 / 10000,
    sample.B05 / 10000,
    sample.B06 / 10000,
    sample.B07 / 10000,
    sample.B08 / 10000,
    sample.B8A / 10000,
    sample.B11 / 10000,
    sample.B12 / 10000,
    sample.SCL
  ];
}
"""

INDICES_AND_SCL = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"],
      units: "DN"
    }],
    output: [
      { id: "indices", bands: 5, sampleType: "FLOAT32" },
      { id: "bands", bands: 4, sampleType: "FLOAT32" },
      { id: "scl", bands: 1, sampleType: "UINT8" }
    ]
  };
}

function evaluatePixel(s) {
  let b2 = s.B02/10000, b3 = s.B03/10000, b4 = s.B04/10000;
  let b8 = s.B08/10000, b11 = s.B11/10000, b12 = s.B12/10000;

  let ndvi = (b8-b4)/(b8+b4+1e-10);
  let ndwi = (b3-b8)/(b3+b8+1e-10);
  let ndmi = (b8-b11)/(b8+b11+1e-10);
  let bsi  = ((b11+b4)-(b8+b2))/((b11+b4)+(b8+b2)+1e-10);
  let msi  = b11/(b8+1e-10);

  return {
    indices: [ndvi, ndwi, ndmi, bsi, msi],
    bands: [b2, b3, b4, b8],
    scl: [s.SCL]
  };
}
"""

# V4 evalscript: returns a full downstream-compatible contract as a single
# default raster response so the Process API request body and the runtime
# GeoTIFF parser stay aligned.
# Output layout:
# [0-7]   indices: NDVI, NDWI, NDMI, BSI, MSI, NDRE, CIre, RE_SLOPE
# [8-17]  raw bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
# [18]    SCL
BANDS_V4_REDEDGE = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"],
      units: "DN"
    }],
    output: { id: "default", bands: 19, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(s) {
  let b2 = s.B02/10000, b3 = s.B03/10000, b4 = s.B04/10000;
  let b5 = s.B05/10000, b6 = s.B06/10000, b7 = s.B07/10000;
  let b8 = s.B08/10000, b8a = s.B8A/10000;
  let b11 = s.B11/10000, b12 = s.B12/10000;

  let ndvi = (b8-b4)/(b8+b4+1e-10);
  let ndwi = (b3-b8)/(b3+b8+1e-10);
  let ndmi = (b8-b11)/(b8+b11+1e-10);
  let bsi  = ((b11+b4)-(b8+b2))/((b11+b4)+(b8+b2)+1e-10);
  let msi  = b11/(b8+1e-10);

  // Red-edge indices for better crop/boundary discrimination
  let ndre  = (b8a-b5)/(b8a+b5+1e-10);       // NDRE: narrow-band red-edge NDVI
  let cire  = b8a/(b5+1e-10) - 1.0;           // CIrededge: chlorophyll index red-edge
  let re_slope = (b7-b5)/(b7+b5+1e-10);       // Red-edge slope (B07-B05 normalized)

  return [
    ndvi, ndwi, ndmi, bsi, msi, ndre, cire, re_slope,
    b2, b3, b4, b5, b6, b7, b8, b8a, b11, b12,
    s.SCL
  ];
}
"""
