// STEP 1: Define ROI with correct geometry
var kancheepuram = ee.Geometry.Rectangle([79.9, 12.4, 80.3, 13.0]);
Map.centerObject(kancheepuram, 10);

// STEP 2: Cloud mask for Landsat SR
function maskLandsatSR(img) {
  var qa = img.select('QA_PIXEL');
  return img.updateMask(qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 5).eq(0)));
}

// STEP 3: Load and preprocess Landsat images
var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'];
function getImage(year) {
  var collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(kancheepuram)
    .filterDate(year + '-01-01', year + '-12-31')
    .filter(ee.Filter.lt('CLOUD_COVER', 20))
    .map(maskLandsatSR)
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'], bands)
    .median().clip(kancheepuram);
    return collection;
}
var image2015 = getImage(2015), image2025 = getImage(2025);

// STEP 4: Define training points
var makePoints = function(coords, label) {
  coords = ee.List(coords);
  return ee.FeatureCollection(coords.map(function(pt) {
    pt = ee.List(pt);
    return ee.Feature(ee.Geometry.Point([pt.get(0), pt.get(1)]), {landcover: label});
  }));
};

var trainingPoints = makePoints([[80.00, 12.85], [80.05, 12.84]], 0) // Built-up
  .merge(makePoints([[80.10, 12.75], [80.12, 12.74]], 1))            // Vegetation
  .merge(makePoints([[80.20, 12.60], [80.25, 12.61]], 2))            // Water
  .merge(makePoints([[80.15, 12.90], [80.18, 12.92]], 3));           // Barren

// STEP 5: Train classifier
var classifier = ee.Classifier.smileCart().train({
  features: image2025.sampleRegions({
    collection: trainingPoints,
    properties: ['landcover'],
    scale: 30
  }),
  classProperty: 'landcover',
  inputProperties: bands
});

// STEP 6: Classify images
var classified2015 = image2015.classify(classifier);
var classified2025 = image2025.classify(classifier);

// STEP 7: Visualization
var vis = {min: 0, max: 3, palette: ['red', 'green', 'blue', 'yellow']};
Map.addLayer(classified2015, vis, 'LULC 2015');
Map.addLayer(classified2025, vis, 'LULC 2025');

// STEP 8: Feature layers
Map.addLayer(classified2015.eq(1).selfMask(), {palette: ['green']}, 'Vegetation 2015');
Map.addLayer(classified2025.eq(1).selfMask(), {palette: ['lightgreen']}, 'Vegetation 2025');
Map.addLayer(classified2015.eq(2).selfMask(), {palette: ['blue']}, 'Water 2015');
Map.addLayer(classified2025.eq(2).selfMask(), {palette: ['lightblue']}, 'Water 2025');
Map.addLayer(classified2015.eq(3).selfMask(), {palette: ['yellow']}, 'Barren 2015');
Map.addLayer(classified2025.eq(3).selfMask(), {palette: ['lightyellow']}, 'Barren 2025');

// STEP 9: Change map
var changeMap = classified2025.subtract(classified2015);
Map.addLayer(changeMap, {min: -3, max: 3, palette: ['white', 'black']}, 'Change 2015-2025');

// STEP 10: Trend Analysis (fractions)
function trend(classVal, image) {
  return image.eq(classVal).reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: kancheepuram,
    scale: 30,
  });
}

print('Vegetation 2015 fraction:', trend(1, classified2015));
print('Vegetation 2025 fraction:');
print('Water 2015 fraction:');
print('Water 2025 fraction:', trend(2, classified2025));
print('Barren 2015 fraction:', trend(3, classified2015));
print('Barren 2025 fraction:', trend(3, classified2025));
// STEP 11: Area calculation function (sq. km)
function computeArea(image, classVal, year) {
  var areaImage = image.eq(classVal).multiply(ee.Image.pixelArea()); // in m²
  var area = areaImage.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: kancheepuram,
    scale: 30,
    maxPixels: 1e9
  }).get('classification'); // this matches band name after .classify()

  return ee.Number(area).divide(1e6).round().format('%.0f').cat(' sq.km (Class ' + classVal + ', ' + year + ')');
}

// Print areas for each class and year
print('Built-up 2015:', computeArea(classified2015, 0, 2015));
print('Vegetation 2015:', computeArea(classified2015, 1, 2015));
print('Water 2015:', computeArea(classified2015, 2, 2015));
print('Barren 2015:', computeArea(classified2015, 3, 2015));

print('Built-up 2025:', computeArea(classified2025, 0, 2025));
print('Vegetation 2025:', computeArea(classified2025, 1, 2025));
print('Water 2025:', computeArea(classified2025, 2, 2025));
print('Barren 2025:', computeArea(classified2025, 3, 2025));
