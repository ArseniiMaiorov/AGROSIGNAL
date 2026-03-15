S1_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VV", "VH"] }],
    output: { id: "default", bands: 2, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  return [sample.VV, sample.VH];
}
"""
