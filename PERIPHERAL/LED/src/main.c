#include <stdio.h>
#include <stdint.h>
#include "pca9632.h"

int main(int argc, char *argv[]) {
    uint8_t led_r, led_g, led_b;

    if (argc != 5) {
        puts("Usage: ./led-example <I2C_DEV_NAME> <R> <G> <B>");
        return -1;
    }

    if (pca9632_init(argv[1]) != 0) {
        perror("pca9632_init failed");
        return -1;
    }

    led_r = atoi(argv[2]);
    led_g = atoi(argv[3]);
    led_b = atoi(argv[4]);

    pca9632_setLED(led_r, led_g, led_b);
    return 0;
}