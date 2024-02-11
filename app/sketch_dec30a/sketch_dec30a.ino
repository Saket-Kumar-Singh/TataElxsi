// Arduino code for a car with 8 ultrasonic sensors for front 180-degree view


#define motorLeft 12
#define motorRight 13


int trigPins[] = {2, 3, 4, 5, 6, 7, 8, 9};
int echoPins[] =  {A0, A1, A2, A3, A4, A5, 10, 11};

long measureDistance(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  return pulseIn(echoPin, HIGH) * 0.034 / 2;
}

void setup() {
  Serial.begin(9600);

  // Initialize ultrasonic sensor pins
  for (int i = 0; i < 8; i++) {
    pinMode(trigPins[i], OUTPUT);
    pinMode(echoPins[i], INPUT);
  }

  // Initialize motor control pins
  pinMode(motorLeft, OUTPUT);
  pinMode(motorRight, OUTPUT);
}

void loop() {
  // Measure distances from each sensor
  long distances[8];
  for (int i = 0; i < 8; i++) {
    distances[i] = measureDistance(trigPins[i], echoPins[i]);
  }

  // Print distance readings
  Serial.print("Distances: ");
  for (int i = 0; i < 8; i++) {
    Serial.print(distances[i]);
    Serial.print(" cm | ");
  }
  Serial.println();

  // Obstacle avoidance logic
  bool obstacleRight = false;
  bool obstacleLeft = false;

  for (int i = 0; i < 4; i++) {
    if (distances[i] < 20) {
      obstacleRight = true;
      break;
    }
  }

  for (int i = 4; i < 8; i++) {
    if (distances[i] < 20) {
      obstacleLeft = true;
      break;
    }
  }
//
//  // Motor control based on obstacle detection
//  if (obstacleRight) {
//    // Obstacle detected on the right, turn left
//    digitalWrite(motorLeft, LOW);
//    digitalWrite(motorRight, HIGH);
//  } else if (obstacleLeft) {
//    // Obstacle detected on the left, turn right
//    digitalWrite(motorLeft, HIGH);
//    digitalWrite(motorRight, LOW);
//  } else {
//    // No obstacle detected, move forward
//    digitalWrite(motorLeft, HIGH);
//    digitalWrite(motorRight, HIGH);
//  }

  
  delay(100); // Adjust the delay based on your application requirements
}
