#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// scarry null pointer!!!
char *waydroid_host = NULL;
char *app = NULL;

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
    //printf("%s\n", waydroid_host);
    system(cmd);
}


void waydroid_disconnect() {
    printf("Disconnecting from waydroid\n");
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "adb disconnect %s", waydroid_host);
    system(cmd);
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

// void get_block() {
//     FILE *fp = popen("python3 utils/waydroid/vision/get_block.py", "r");
//     if (!fp) {
//         perror("popen");
//         exit(1);
//     }
//
//     // FILE *out = fopen("out", "wb");
//     // if (!out) {
//     //     perror("fopen");
//     //     exit(1);
//     // }
//     int c;
//     //size_t n;
//     size_t i = 0;
//
//     //read first 6 bytes to get sizes
//     unsigned char sizes[6];
//
//     // Read bytes until HEADER_MARKER (0xFD) is reached
//     //should've read the book...
//     while ((c = fgetc(fp)) != EOF) {
//         if ((unsigned char)c == 0xFD) break;  // end of header
//
//         sizes[i++] = (unsigned char)c;
//
//         if (i >= 6) {
//             fprintf(stderr, "Unexpected more than 6 header bytes!\n");
//             break;
//         }
//     }
//
//     int num_blocks = (int)i / 2;  // int i is number of bytes read from header
//     if (num_blocks > 3) num_blocks = 3; // in case your array only supports 3 blocks
//
//     // // print sizes
//     // for (int j = 0; j < i; j += 2) {
//     //     printf("Block %d rows: %02x\n", j/2, sizes[j]);
//     //     printf("Block %d cols: %02x\n", j/2, sizes[j+1]);
//     // }
//
//     // while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
//     //
//     // }
//
//     /*
//      *
//      *    0x000000: 0x01 0xfe 0x01 0xfe 0x01 0xfe 0x01 0xfe 0xff 0x01 ..........
//      *    0x00000a: 0xfe 0x01 0xfe 0x01 0xfe 0x01 0xfe 0xff 0x01 0x01 ..........
//      *    0x000014: 0x01 0xfe 0x00 0x01 0x00 0xfe 0xff                .......
//      *    bytes: 27
//      *
//      */
//
//     int row = 0, col = 0, block = 0;
//     while ((c = fgetc(fp)) != EOF) {
//         unsigned char byte = (unsigned char)c;
//         if (byte == 0xFE) { //EOR - End Of Row
//             row++; // go down one row and go at the start
//             col = 0;
//         }else if (byte == 0xFF) { //end of block
//             block++; // go to the next block and reset
//             row = 0;
//             col = 0;
//         }else {
//             blocks[block].pattern[row][col++] = byte;
//         }
//     }
//
//     pclose(fp);
//
//     for (int b = 0; b < num_blocks; b++) {
//
//         printf("Block %d:\n", b);
//         for (int r = 0; r < sizes[b*2]; r++) {      // use sizes array for row count
//             for (int d = 0; d < sizes[b*2 + 1]; d++) { // and for col count
//                 printf("%d", blocks[b].pattern[r][d]);
//             }
//             printf("\n");
//         }
//         printf("\n");
//     }
//
// }

// void get_grid(int grid[8][8]) {
//     FILE *fp = popen("python3 utils/waydroid/vision/get_grid.py", "r");
//     if (!fp) {
//         perror("popen");
//         exit(1);
//     }
//
//     // FILE *out = fopen("out", "wb");
//     // if (!out) {
//     //     perror("fopen");
//     //     exit(1);
//     // }
//     //
//     // size_t n;
//     //
//     // char buff[256];
//     // while ((n = fread(buff, 1, sizeof(buff), fp)) > 0) {
//     //     fwrite(buff, 1, n, out);
//     // }
//
//     int c;
//
//     int row = 0, col = 0;
//     while ((c = fgetc(fp)) != EOF) {
//         unsigned char byte = (unsigned char)c;
//         if (byte == 0xFE) { //EOR - End Of Row
//             row++; // go down one row and go at the start
//             col = 0;
//         }else {
//             grid[row][col++] = byte;
//         }
//     }
// }

// int coords[3] = {0};
//
// void get_block_coord() {
//     FILE *fp = popen("python3 utils/waydroid/vision/get_block_coord.py", "r");
//
//     if (!fp) {
//         perror("popen");
//         exit(1);
//     }
//
//     // FILE *out = fopen("out", "wb");
//     // if (!out) {
//     //     perror("fopen");
//     //     exit(1);
//     // }
//
//     /*Example output:
//      *
//      * 0x000000: 0x03 0x03 0x02 0xee 0xff 0x04 0x21 0x02 0xee 0xff ......!...
//      * 0x00000a: 0x03 0x9e 0x02 0xfa 0xff                          .....
//      */
//
//     //size_t n;
//     int c;
//
//     // char buff[256];
//     // while ((n = fread(buff, 1, sizeof(buff), fp)) > 0) {
//     //     fwrite(buff, 1, n, out);
//     // }
//
//     int coord_index = 0;
//     unsigned char bytes[5];
//     int count = 0;
//
//     while ((c = fgetc(fp)) != EOF) { // read bytes
//         bytes[count++] = (unsigned char)c;
//         if (count == 5) { // once 5 bytes collected
//         // 0-1 = X (big endian)
//         // 2-3 = Y (big endian)
//         // 4 = marker (0xFF)
//             if (bytes[4] != 0xFF) {
//                 fprintf(stderr, "Marker missing");
//                 exit(1);
//             }
//
//             // combine X and Y into 16-bit ints
//             // because the data is big endian, shift the first byte left by 8 and OR it with the second one
//             int x = (bytes[0] << 8) | bytes[1];
//             int y = (bytes[2] << 8) | bytes[3];
//
//             // store coords in array
//             coords[coord_index++] = x;
//             coords[coord_index++] = y;
//
//             // reset counter for next pair
//             count = 0;
//         }
//     }
