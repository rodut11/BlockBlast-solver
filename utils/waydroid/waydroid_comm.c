#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
//#define STB_IMAGE_IMPLEMENTATION
//#include "../../include/stb_image.h"

// scarry null pointer!!!
char *waydroid_host = NULL;

void waydroid_connect(const char *host) {
    // free old one if it exists
    if (waydroid_host) free(waydroid_host);

    // allocate memory for the copy (+1 for null terminator)
    waydroid_host = malloc(strlen(host) + 1);
    if (!waydroid_host) {
        fprintf(stderr, "Memory allocation failed!\n");
        return;
    }

    // copy the string
    strcpy(waydroid_host, host);

    char cmd[256];
    snprintf(cmd, sizeof(cmd), "adb connect %s &", waydroid_host);
    system(cmd);
}


void waydroid_disconnect() {
    printf("Disconnecting from waydroid\n");
    system("adb disconnect &");
}

unsigned char* get_screencap(size_t* out_size) {
    const size_t chunk_size = 4096;  // read in 4KB chunks
    size_t capacity = chunk_size;
    size_t bytes_read = 0;

    // get image
    FILE* pipe = popen("adb exec-out screencap -p", "r");
    if (!pipe) {
        perror("popen");
        exit(1);
    }

    // check for error
    unsigned char* buffer = malloc(capacity);
    if (!buffer) {
        perror("malloc");
        pclose(pipe);
        exit(1);
    }

    // infinite loop
    while (1) {

        // if the capacity is smaller than the already read bytes and the size of the chunk, double the capacity
        if (bytes_read + chunk_size > capacity) {
            capacity *= 2;

            // craete temp memory and realloc
            unsigned char* tmp = realloc(buffer, capacity);

            //check for erro
            if (!tmp) {
                perror("realloc");
                free(buffer);
                pclose(pipe);
                exit(1);
            }

            //write what's in the temp memory to the buffer
            buffer = tmp;
        }

        // write to the buffer the bytes read
        size_t n = fread(buffer + bytes_read, 1, chunk_size, pipe);


        // if fread() returns 0 (reached the end of stream)
        if (n == 0) {
            if (feof(pipe)) break;  // end of stream

            // check for error
            if (ferror(pipe)) {
                perror("fread");
                free(buffer);
                pclose(pipe);
                exit(1);
            }
        }
        bytes_read += n;
    }

    pclose(pipe);
    *out_size = bytes_read;
    return buffer;
}
