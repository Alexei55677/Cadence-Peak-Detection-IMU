#include <Arduino.h>
#include <Arduino_BMI270_BMM150.h>

float ax = 0.0, ay = 0.0, az = 0.0;
float gx = 0.0, gy = 0.0, gz = 0.0;

unsigned long sampleIndex = 0;


void setup() {

    Serial.begin(115200);
    while (!Serial);

    if (!IMU.begin()) {
        Serial.println("Error: IMU failed to initialize.");
        while (1);
    }

    IMU.setContinuousMode();

    Serial.println(
        "sample\tax_g\tay_g\taz_g\tgx_dps\tgy_dps\tgz_dps"
    );

}

void loop() {

    if (IMU.accelerationAvailable() &&
        IMU.gyroscopeAvailable()) {

        IMU.readAcceleration(ax, ay, az);
        IMU.readGyroscope(gx, gy, gz);

        Serial.print(sampleIndex);
        Serial.print('\t');

        Serial.print(ax);
        Serial.print('\t');
        Serial.print(ay);
        Serial.print('\t');
        Serial.print(az);
        Serial.print('\t');

        Serial.print(gx);
        Serial.print('\t');
        Serial.print(gy);
        Serial.print('\t');
        Serial.println(gz);

        sampleIndex++;
    }

}