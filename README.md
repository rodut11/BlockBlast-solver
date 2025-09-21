# Block Blast Auto Player (C Project)

This is a C project where I’m building a program to automatically play Block Blast, with the goal of getting really high scores.  
The project is still in early stages, but the core idea is to automatically play the game.

---

## Features (so far)

- **Block Representation**  
  Each block is defined by:
  - Width & height
  - A pivot point (centerX, centerY)
  - A pattern (up to 4x4) describing the shape

- **Block Library**  
  Defined in `blocks.c` and exposed via `blocks.h`.  
  Includes:
  - All the classic Block Blast pieces
  - Different rotations
  - ~32 unique block variants

- **Grid System**  
  - The game grid is represented as a 2D array (`grid[MAX_GRID_HEIGHT][MAX_GRID_WIDTH]`)  
  - Functions to print the grid to console (`print_grid`)  
  - Functions to print blocks (`print_block`)

- **Block Placement**  
  - Function `place_block()` places a block at a given grid coordinate  
  - Uses the block’s pivot to calculate correct positioning  
  - Handles grid bounds so blocks don’t overflow outside

---

## Project Structure

```
.
├── CMakeLists.txt
├── include
│   └── ANSI-Color-Codes.h
├── LICENSE
├── README.md
├── src
│   ├── blocks.c
│   ├── blocks.h
│   └── main.c
└── utils
    ├── debug
    │   ├── debug_blocks.h
    │   └── debug_print_blocks.c
    ├── util_blocks.c
    └── util_blocks.h
```

---
## Roadmap

- [x] Define basic blocks  
- [x] Implement rotations  
- [x] Grid + block placement  
- [ ] Line clearing logic  
- [ ] Automatic block selection & placement strategy  
- [ ] Score calculation

---

## Testing

Right now, testing is manual (debug programs + printing grid state).  
Planned: add unit tests (probably with a simple test runner in C).

---

## Dev Logs

You can follow the progress on my **[blog](https://blog.rodut11.dev)**

- **Day 1** → Set up environment, created Block struct, basic grid & printing  
- **Day 2** → Added rotations, wrote `place_block()` logic, debug utilities  

---

## License

This project will be licensed under GPLv3 (FOSS friendly).

---

## Credits

`ANSI-Color-codes.h` - [RabaDabaDoba on Github Gist](https://github.com/RabaDabaDoba) - [Link to gist](https://gist.github.com/RabaDabaDoba/145049536f815903c79944599c6f952a)