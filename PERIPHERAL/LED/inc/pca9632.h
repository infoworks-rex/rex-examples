#ifndef __PCA9632_H__
#define __PCA9632_H__

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

//PCA9632 Sensor Slave Address
#define PCA9632_I2C_ADDRESS 0x70

//PCA9632 Sensor Mode Set
#define PCA9632_MODE 0x00

//PCA9632 Sensor Register Address
#define PCA9632_LEDR_PWM 0x02
#define PCA9632_LEDG_PWM 0x04
#define PCA9632_LEDB_PWM 0x03
#define PCA9632_LED_OUT 0x08
#define PCA9632_RESET 0x06

// Returns file descriptor
int pca9632_init(char* dev_name);
int pca9632_setLED(uint8_t led_r, uint8_t led_g, uint8_t led_b);

#endif

