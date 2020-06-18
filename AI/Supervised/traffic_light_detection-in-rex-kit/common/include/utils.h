#pragma once
#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdio.h>
#include <stdint.h>

#define SET_CONSOLE_RESET         printf("\033[0m");

#define SET_CONSOLE_RED           printf("\033[0;31m");
#define SET_CONSOLE_RED_BOLD      printf("\033[1;31m");
#define SET_CONSOLE_GREEN         printf("\033[0;32m");
#define SET_CONSOLE_GREEN_BOLD    printf("\033[1;32m");
#define SET_CONSOLE_YELLOW        printf("\033[0;33m");
#define SET_CONSOLE_YELLOW_BOLD   printf("\033[01;33m");
#define SET_CONSOLE_BLUE          printf("\033[0;34m");
#define SET_CONSOLE_BLUE_BOLD     printf("\033[1;34m");
#define SET_CONSOLE_MAGENTA       printf("\033[0;35m");
#define SET_CONSOLE_MAGENTA_BOLD  printf("\033[1;35m");
#define SET_CONSOLE_CYAN          printf("\033[0;36m");
#define SET_CONSOLE_CYAN_BOLD     printf("\033[1;36m");


/**
* @brief Make dummy function and print todo message
*
* @param _type		Function return type
* @param _name		Dummy function name
* @param _content	Dummy function content(printf, return, etc...)
* @param _args... Dummy function parameters
*
* @return Return value is depends on "_content"
*/
#define DUMMY_FUNC(_type, _name, _content, _args...) \
_type _name(_args); \
_type _name(_args) \
{ \
   printf("TODO: '%s %s(%s)' at '%s:%d'\n", \
          #_type, #_name, #_args, __FILE__, __LINE__); \
   _content; \
}

/**
* @brief Print ToDo message
*
* @param fmt		ToDo contents
* @param args...	ToDo args
*
* @return 
*/
#define TODO(fmt, args...) \
		do {	\
			printf("[%s:%d:%s()] TODO : " fmt, \
			__FILE__, __LINE__, __func__, ##args); \
		}while(0)

/**
* @brief Print error message to stderr
*
* @param x			Error message
* @param arg...	args
*
* @return Nothing
*/
#define ERR_MSG(x,arg...)			\
		do{								\
			SET_CONSOLE_GREEN_BOLD;		\
			fprintf(stderr, x,##arg);	\
			SET_CONSOLE_RESET;			\
		}while(0)

/**
* @brief Print debug message to stderr
*
* @param x			Debug message
* @param arg...	args
*
* @return Nothing
*/
#define DBG_MSG(x,arg...)					\
		do {										\
			SET_CONSOLE_GREEN_BOLD;			\
			fprintf(stderr, x,##arg);		\
			SET_CONSOLE_RESET;				\
		} while(0)

#define CLEAR(x) memset (&(x), 0, sizeof(x))

#endif
