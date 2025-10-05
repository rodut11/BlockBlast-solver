#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
//#define STB_IMAGE_IMPLEMENTATION
#include <stdint.h>

#include "../../../src/blocks.h"

// scarry null pointer!!!
char *waydroid_host = NULL;
char *app = NULL;

undefined_block blocks[3] = {0};

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

            // create temp memory and realloc
            unsigned char* tmp = realloc(buffer, capacity);

            //check for error
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

void open_app(const char *input_app) {

    if (app) free(app);

    app = malloc(strlen(input_app) + 1);
    if (!app) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }

    char cmd[256];
    strcpy(app, input_app);

    snprintf(cmd, sizeof(cmd), "adb shell am start -n %s", app);
    system(cmd);
}

void get_block() {
    FILE *fp = popen("python3 /home/rodut11/Coding/C/Block-blast-solver/utils/waydroid/python/getblock.py", "r");
    if (!fp) {
        perror("popen");
        exit(1);
    }

    // FILE *out = fopen("out", "wb");
    // if (!out) {
    //     perror("fopen");
    //     exit(1);
    // }

    char buf[128];
    int c;
    size_t n;
    size_t i = 0;

    //read first 6 bytes to get sizes
    unsigned char sizes[6];

    // Read bytes until HEADER_MARKER (0xFD) is reached
    //should've read the book...
    while ((c = fgetc(fp)) != EOF) {
        if ((unsigned char)c == 0xFD) break;  // end of header

        sizes[i++] = (unsigned char)c;

        if (i >6) {
            fprintf(stderr, "Unexpected more than 6 header bytes!\n");
            break;
        }
    }


    // // print sizes
    // for (int j = 0; j < i; j += 2) {
    //     printf("Block %d rows: %02x\n", j/2, sizes[j]);
    //     printf("Block %d cols: %02x\n", j/2, sizes[j+1]);
    // }

    // while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
    //
    // }

    /*
     *
     *    0x000000: 0x01 0xfe 0x01 0xfe 0x01 0xfe 0x01 0xfe 0xff 0x01 ..........
     *    0x00000a: 0xfe 0x01 0xfe 0x01 0xfe 0x01 0xfe 0xff 0x01 0x01 ..........
     *    0x000014: 0x01 0xfe 0x00 0x01 0x00 0xfe 0xff                .......
     *    bytes: 27
     *
     */

    int row = 0, col = 0, block = 0;
    while ((c = fgetc(fp)) != EOF) {
        unsigned char byte = (unsigned char)c;
        if (byte == 0xFE) { //EOR - End Of Row
            row++; // go down one row and go at the start
            col = 0;
        }else if (byte == 0xFF) { //end of block
            block++; // go to the next block and reset
            row = 0;
            col = 0;
        }else {
            blocks[block].pattern[row][col++] = byte;
        }
    }

    pclose(fp);

    for (int b = 0; b < 3; b++) {
        printf("Block %d:\n", b);
        for (int r = 0; r < sizes[b*2]; r++) {      // use sizes array for row count
            for (int d = 0; d < sizes[b*2 + 1]; d++) { // and for col count
                printf("%d", blocks[b].pattern[r][d]);
            }
            printf("\n");
        }
        printf("\n");
    }

}