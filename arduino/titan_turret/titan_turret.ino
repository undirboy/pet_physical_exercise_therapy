#include <Servo.h>

// Hardware Objects
Servo panServo;       // Aims the turret left/right
Servo feederServo;    // Pushes the treat into the launcher

// Pin Definitions
const int panPin = 9;
const int feederPin = 11;
const int launcherMotorPin = 10; // Must be a PWM (~) pin for speed control

void setup() {
  Serial.begin(9600); // Matches the 9600 baud rate in your Python script
  
  panServo.attach(panPin);
  feederServo.attach(feederPin);
  pinMode(launcherMotorPin, OUTPUT);

  // Set default starting positions
  panServo.write(90);       // Center the turret
  feederServo.write(0);     // Retract the feeder arm
  analogWrite(launcherMotorPin, 0); // Flywheels off
}

void loop() {
  // Listen for the string from Python over the USB cable
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    // Check if the message is our trigger command
    if (command.startsWith("FIRE:")) {
      
      // Chop up the string to find the two numbers
      int firstColon = command.indexOf(':');
      int secondColon = command.indexOf(':', firstColon + 1);

      if (firstColon != -1 && secondColon != -1) {
        String angleStr = command.substring(firstColon + 1, secondColon);
        String distStr = command.substring(secondColon + 1);

        int targetAngle = angleStr.toInt();
        int targetDistance = distStr.toInt();

        // Execute the physical launch sequence
        fireTreat(targetAngle, targetDistance);
      }
    }
  }
}

void fireTreat(int angle, int distance) {
  // 1. Aim the turret base
  panServo.write(angle);
  delay(600); // Give the base time to rotate into position

  // 2. Spin up the launcher motors
  // This maps your UI distance (3 to 10 feet) to a motor PWM speed (150 to 255)
  int motorSpeed = map(distance, 3, 10, 150, 255);
  motorSpeed = constrain(motorSpeed, 100, 255); // Safety limit
  
  analogWrite(launcherMotorPin, motorSpeed);
  delay(1000); // Wait 1 second for flywheels to reach max RPM

  // 3. Push the treat into the spinning flywheels
  feederServo.write(90); // Flick the arm forward
  delay(400);
  feederServo.write(0);  // Instantly pull the arm back
  delay(400);

  // 4. Spin down motors and return to standby
  analogWrite(launcherMotorPin, 0);
  panServo.write(90);
}
