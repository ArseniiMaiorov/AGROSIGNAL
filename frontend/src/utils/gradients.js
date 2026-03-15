export const LAYER_GRADIENTS = {
  ndvi: {
    colors: ['#6b1d1d', '#a53b1f', '#d1a536', '#7fbf58', '#3a8b46', '#1f4d32'],
    range: [0, 1],
    unit: '',
    format: (value) => Number(value).toFixed(2),
  },
  ndwi: {
    colors: ['#7a4b20', '#c17d3b', '#d6c16e', '#7ab7d8', '#2d78c4', '#173f7a'],
    range: [-1, 1],
    unit: '',
    format: (value) => Number(value).toFixed(2),
  },
  ndmi: {
    colors: ['#8c5a2b', '#b67c3b', '#cbb168', '#7fb36c', '#3d8b59', '#1d5a41'],
    range: [-1, 1],
    unit: '',
    format: (value) => Number(value).toFixed(2),
  },
  bsi: {
    colors: ['#234d2d', '#5e8c4d', '#b0b76b', '#d7b066', '#bd6f3a', '#7f3420'],
    range: [-1, 1],
    unit: '',
    format: (value) => Number(value).toFixed(2),
  },
  precipitation: {
    colors: ['#f2f4f7', '#b8d4e3', '#6aa4ca', '#2f70b7', '#183f7a'],
    range: [0, 100],
    unit: 'мм',
    format: (value) => `${Number(value).toFixed(0)} мм`,
  },
  wind: {
    colors: ['#f7f9fb', '#c6d7e8', '#87abd0', '#4774aa', '#26456f'],
    range: [0, 30],
    unit: 'м/с',
    format: (value) => `${Number(value).toFixed(1)} м/с`,
  },
  soil_moisture: {
    colors: ['#d9b65d', '#bcc96c', '#7fa85c', '#3f7d6b', '#215570'],
    range: [0, 1],
    unit: '',
    format: (value) => `${(Number(value) * 100).toFixed(0)}%`,
  },
  gdd: {
    colors: ['#f7f0d7', '#e9d89b', '#dba15f', '#b55a2f', '#703018'],
    range: [0, 3000],
    unit: '°С·дн',
    format: (value) => `${Number(value).toFixed(0)} °С·дн`,
  },
  vpd: {
    colors: ['#e7eef4', '#bed0dc', '#7ea3bc', '#4f6f9c', '#263c63'],
    range: [0, 4],
    unit: 'кПа',
    format: (value) => `${Number(value).toFixed(2)} кПа`,
  },
}

export function getGradientForLayer(layerId) {
  return LAYER_GRADIENTS[layerId] || LAYER_GRADIENTS.ndvi
}

export function gradientCss(layerId) {
  const gradient = getGradientForLayer(layerId)
  return `linear-gradient(90deg, ${gradient.colors.join(', ')})`
}

export function valueToColor(value, layerId) {
  const gradient = getGradientForLayer(layerId)
  const [minValue, maxValue] = gradient.range
  const normalized = clamp((Number(value) - minValue) / Math.max(maxValue - minValue, 1e-6), 0, 1)
  const segment = normalized * (gradient.colors.length - 1)
  const leftIndex = Math.floor(segment)
  const rightIndex = Math.min(gradient.colors.length - 1, leftIndex + 1)
  const ratio = segment - leftIndex
  return interpolateHex(gradient.colors[leftIndex], gradient.colors[rightIndex], ratio)
}

function interpolateHex(leftColor, rightColor, ratio) {
  const left = hexToRgb(leftColor)
  const right = hexToRgb(rightColor)
  return `rgba(${Math.round(left.r + (right.r - left.r) * ratio)}, ${Math.round(left.g + (right.g - left.g) * ratio)}, ${Math.round(left.b + (right.b - left.b) * ratio)}, 0.62)`
}

function hexToRgb(hex) {
  const raw = hex.replace('#', '')
  return {
    r: parseInt(raw.slice(0, 2), 16),
    g: parseInt(raw.slice(2, 4), 16),
    b: parseInt(raw.slice(4, 6), 16),
  }
}

function clamp(value, lower, upper) {
  return Math.max(lower, Math.min(upper, value))
}
