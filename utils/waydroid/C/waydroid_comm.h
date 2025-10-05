//
// Created by rodut11 on 9/26/25.
//

#ifndef C_WAYDROID_COMM_H
#define C_WAYDROID_COMM_H

void waydroid_disconnect();
void waydroid_connect(const char *host);
unsigned char* get_screencap(size_t* out_size);
void open_app(const char *input_app);

#endif //C_WAYDROID_COMM_H