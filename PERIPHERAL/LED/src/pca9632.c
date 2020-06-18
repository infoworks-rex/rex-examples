#include "pca9632.h"

#include <stdio.h>
#include <stdlib.h>
#include <linux/i2c-dev.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <string.h>
#include <pthread.h>

static int fd;

//PCA9632 Sensor Register Data Write
static int WriteData(uint8_t reg_addr, uint8_t *data, int size) {
    uint8_t *buf;

    buf = malloc(size + 1);
    buf[0] = reg_addr;
    memcpy(buf + 1, data, size);
    write(fd, buf, size + 1);
    free(buf);

    return 0;
}

//-----------------------------------------------------------

//PCA9632 Sensor Register Data Read
static int ReadData(uint8_t reg_addr, uint8_t *data, int size) {
    write(fd, &reg_addr, 1);
    read(fd, data, size);

    return 0;
}

//-----------------------------------------------------------

//PCA9632 Sensor Check
static void I2C_Slave_Check(void)
{	
	if(ioctl(fd, I2C_SLAVE, PCA9632_I2C_ADDRESS) < 0)
	{
		printf("Failed to acquire bus access and/or talk to slave\n");
		exit(1);
	}

	if(write(fd, NULL, 0) < 0)
	{
		printf("[PCA9632(0x70)] I2C Sensor Is Missing\n");
		exit(1);
	}
	else
	{
//		printf("Check OK!! [PCA9632(0x70)] I2C Sensor\n");
	}
}

//-----------------------------------------------------------

//PCA9632 Sensor Reset
static void Reset(void)
{
	uint8_t data[3] = {0xA5, 0x5A};

	WriteData(PCA9632_RESET, data, 2);

	//printf("PCA9632 Reset Finish\n");
}

//-----------------------------------------------------------

//PCA9632 Sensor Mode Setting
static void Init(void) 
{
	uint8_t data = 0x0f;

	WriteData(PCA9632_MODE, &data, 1);

	//printf("PCA9632 Mode Setting Finish\n");
}

//-----------------------------------------------------------

//PCA9632 LEDOut Setting
static void LEDOUT_Setting(void)
{
	uint8_t data = 0xff;

	WriteData(PCA9632_LED_OUT, &data, 1);
	//printf("PCA9632 LEDOut Setting Finish\n");
}

//-----------------------------------------------------------

//PCA9632 First_Register_Setting
static void First_Register_Setting(void)
{
	uint8_t data = 0;

	WriteData(PCA9632_LEDR_PWM, &data, 1);
	WriteData(PCA9632_LEDG_PWM, &data, 1);
	WriteData(PCA9632_LEDB_PWM, &data, 1);
	//printf("PCA9632 First_Register_Setting Finish\n");
}

//-----------------------------------------------------------

int pca9632_init(char *dev_name) {
	if ((fd = open(dev_name, O_RDWR)) < 0) {
		perror("Failed to open pca9632");
		return -1;
	}

	I2C_Slave_Check();
	Reset();
	Init();
	LEDOUT_Setting();
	First_Register_Setting();

	return 0;
}

int pca9632_setLED(uint8_t led_r, uint8_t led_g, uint8_t led_b) {
	WriteData(PCA9632_LEDR_PWM, &led_r, 1);
	WriteData(PCA9632_LEDG_PWM, &led_g, 1);
	WriteData(PCA9632_LEDB_PWM, &led_b, 1);

	return 0;
}
