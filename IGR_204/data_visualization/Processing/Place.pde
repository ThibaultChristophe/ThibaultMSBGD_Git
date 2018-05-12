class Place implements Comparable {
  float longitude;
  float latitude;
  String name;
  int postalCode;
  float population;
  float density;
  
  int x() {
    return int(map(this.longitude, minX, maxX, 20, width-20));
  }
  
  int y() {
    return int(map(this.latitude, minY, maxY, height-20, 20));
  }
  

  float ln(float d, float t) {
    return log(t+d);
  }
  
  int density() {
    float t = 100.;
    return int(map(
      ln(density, t), 
      ln(minDensity, t), 
      ln(maxDensity, t), 
      0, 255));
    //return int(map(density, minDensity, maxDensity, 0, 255));
  }
  
  int population() {
    float t = 400.;
    return int(map(
      ln(population, t), 
      ln(minPop, t), 
      ln(maxPop, t), 
      1., 24.));
    //return int(map(population, minPop, maxPop, 2., 124.));
  }
  
  void draw() {
    int d = density();
    int p = population();
    fill(255, 255-d, 0, 180);
    ellipse(x(), y(), p, p);
  }
  
  String label() {
    return name +
      "\nPopulation: " + population +
      "\nDensity: " + density +
      "\nPostal code: " + postalCode;
  }
  
  int compareTo(Object other) {
    return int(population - ((Place)other).population);
  }
}