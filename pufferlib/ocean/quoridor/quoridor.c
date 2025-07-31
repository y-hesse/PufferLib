#include <time.h>
#include <cstdio>
#include "quoridor.h"
#include "raylib.h"

#define BOARD_SIZE WALLS

void demo() {
    Quoridor env;
    float observations[(WALLS * WALLS + 3)*2] = {0};
    int actions[2] = {0};
    float rewards[2] = {0};
    unsigned char terminals[2] = {0};

    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.moves = 0;

    c_reset(&env);

    const int px = 40;
    const int board_px = WALLS * px;
    InitWindow(board_px + 200, board_px, "Quoridor RL");
    SetTargetFPS(60);
    int total_actions = (RIGHT+1) + 2*(SIZE-1)*(SIZE-1);

    int i = 0;
    int wall_id = 0;
    while (!WindowShouldClose() && i < 100000000) {
        int p1_action = -1;
        // Wait for player input before next action
        if (true) {
            bool player_moved = false;
            if (env.players[1].is_turn) {
                while (!player_moved && !WindowShouldClose()) {
                    c_render(&env);
                    if (IsKeyPressed(KEY_UP)) { p1_action = UP; player_moved = true; }
                    else if (IsKeyPressed(KEY_DOWN)) { p1_action = DOWN; player_moved = true; }
                    else if (IsKeyPressed(KEY_LEFT)) { p1_action = LEFT; player_moved = true; }
                    else if (IsKeyPressed(KEY_RIGHT)) { p1_action = RIGHT; player_moved = true; }
                    // Optionally add wall placement for player 1
                    if (IsKeyPressed(KEY_RIGHT)) { 
                        p1_action = RIGHT + wall_id + 1 + 0; 
                        wall_id++; 
                        printf("Mouse pressed");
                    }
                    // if (IsKeyPressed(KEY_V)) { p1_action = RIGHT + 1 + (SIZE-1); player_moved = true; }
                    if (IsKeyPressed(KEY_ESCAPE)) { CloseWindow(); exit(0); }
                    WaitTime(0.01);
                }
                if (p1_action == -1) p1_action = 0; // Default to UP if no input (shouldn't happen)
                    env.actions[1] = p1_action;
                } else {
                    // Not player 1's turn, set NOOP
                    env.actions[1] = 0; // UP as NOOP (or define a NOOP constant if you want)
                }
        }
        //env.actions[0] = rand() % total_actions;
        env.actions[0] = rand() % 4;
        c_step(&env);
        i+=1;
        //c_render(&env);
    }
    // No freeing, let OS reclaim memory on exit
}

int main() {
    srand(time(NULL));
    demo();
    return 0;
}