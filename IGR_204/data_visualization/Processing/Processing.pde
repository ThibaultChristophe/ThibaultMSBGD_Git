import java.util.*;

ArrayList<Float> x, y, populations, densities;
ArrayList<Place> places;
float minX, maxX;
float minY, maxY;
float minPop, maxPop;
float minDensity, maxDensity;
Place closestPlace = null;
PFont labelFont = null;

void setup() {
  size(840, 840);
  noLoop();
  readData();
  labelFont = loadFont("Lato-Regular-28.vlw");
  textFont(labelFont, 28);
}

void draw() {
  //background(135,206,235);
  background(40,70,128);
  noStroke();
  for (Place place: places) {
    place.draw();
  }
  
  drawLegend();
  drawLabel();
}

void mouseMoved() {
  float dist, minDist = 99999999.;
  int dx, dy;
  for (Place place: places) {
    dx = mouseX - int(place.x());
    dy = mouseY - int(place.y());
    dist = dx*dx + dy*dy;
    if (dist < minDist) {
      closestPlace = place;
      minDist = dist;
    }
  }
  
  if (minDist > 30) {
    closestPlace = null;
    return;
  }
  
  redraw();
}

void drawLabel() {
  if (closestPlace == null) 
    return;

  try {
    fill(255);
    stroke(0, 255, 0);
    strokeWeight(3);
    int posX = closestPlace.x();
    int posY = closestPlace.y();
    
    int crossSize = 14;
    line(posX-crossSize, posY, posX+crossSize, posY);
    line(posX, posY-crossSize, posX, posY+crossSize);
    text(closestPlace.label(), 10, height-120);
  } catch (Exception ex) {}
}


void drawLegend() {
  for (int i = 0; i < 255; i++) {
    fill(255, 255-i, 0);
    rect(360 + 0.2 * i, 766, 0.4, 20);
  }
  fill(255);
  Integer minDensityInt = Math.round(minDensity);
  Integer maxDensityInt = Math.round(maxDensity);
  text("Density: "+minDensityInt+" - "+maxDensityInt, 420, 786);
  
  stroke(0);
  strokeWeight(1);
  fill(40,70,128);
  ellipse(360+16, 796+12, 24, 24);
  ellipse(360+12, 796+12, 13, 13);
  ellipse(360+8, 796+12, 3, 3);
  
  fill(255);
  Integer minPopInt = Math.round(minPop);
  Integer maxPopInt = Math.round(maxPop);
  text("Population: "+minPopInt+" - "+maxPopInt, 420, 816);
}


void readData() {
  //String[] lines = loadStrings("https://perso.telecom-paristech.fr/eagan/class/igr204/data/population.tsv");
  String[] lines = loadStrings("population.tsv");
  println("Loaded " + lines.length + " lines.");
  
  int l = lines.length-2;
  x = new ArrayList<Float>();
  y = new ArrayList<Float>();
  populations = new ArrayList<Float>();
  densities = new ArrayList<Float>();
  places = new ArrayList<Place>();
  
  for (int i=0; i < l; i++) {
    String[] columns = lines[i+2].split("\t");
    if (Float.isNaN(float(columns[1])) || Float.isNaN(float(columns[2]))) {
      continue;
    }
    
    Place place = new Place();
    place.postalCode = Integer.parseInt(columns[0]);
    place.longitude = float(columns[1]);
    place.latitude = float(columns[2]);
    place.name = columns[4];
    place.population = Integer.parseInt(columns[5]);
    place.density = Integer.parseInt(columns[6]);
    
    places.add(place);
    
    x.add(place.longitude);
    y.add(place.latitude);
    populations.add(place.population);
    densities.add(place.density);
  }
  
  minX = Collections.min(x);
  maxX = Collections.max(x);
  minY = Collections.min(y);
  maxY = Collections.max(y);
  minPop = Collections.min(populations);
  maxPop = Collections.max(populations);
  minDensity = Collections.min(densities);
  maxDensity = Collections.max(densities);
  
  Collections.sort(places);
}