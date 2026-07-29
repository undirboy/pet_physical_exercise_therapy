# PitPaws Titan Station - Hardware Guide

This folder contains the Arduino C++ firmware required to power the physical motors of the Titan Station.

## How It Works
The Python computer vision backend communicates with the Arduino over USB Serial at a `9600` baud rate. When a pet triggers a launch, Python sends a command formatted as:
`FIRE:[ANGLE]:[DISTANCE]\n` (e.g., `FIRE:90:6\n`).

The Arduino parses this string and translates it into physical motor movements.

## Wiring Guide

| Component | Arduino Pin | Notes |
| :--- | :--- | :--- |
| **Pan Servo** | Digital Pin 9 | Base servo. Rotates the turret 0° to 180°. |
| **Feeder Servo** | Digital Pin 11 | Internal servo. Flicks 90° to push the treat into the flywheels. |
| **Launcher Motors** | Digital Pin 10 (PWM) | Controls the RPM of the dual DC flywheels to adjust launch distance. |

### ⚠️ WARNING: Launcher Motor Power
**Do not** plug the DC launcher motors directly into the Arduino! The current draw will permanently damage the microcontroller. 

You must route Pin 10 through a **Motor Driver (e.g., L298N)** or an N-Channel MOSFET, and power the motors using an external power supply or battery pack. Ensure the external power supply and the Arduino share a common Ground (GND).
