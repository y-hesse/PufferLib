#include <string.h>
#include <math.h>
#include "raylib.h"

#define SIZE 10
#define INIT_WALLS 10

#define EMPTY 0
#define WALL 1
#define WALLS (SIZE*2-1)
#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3


// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;

typedef struct {
    int x;
    int y;
    int walls;
    bool is_turn;
} Player;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    Player players[2];
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int moves;
    int winner;
    float walls[WALLS][WALLS];
} Quoridor;


/* Recommended to have an observation function of some kind because
 * you need to compute agent observations in both reset and in step.
 * If using float obs, try to normalize to roughly -1 to 1 by dividing
 * by an appropriate constant.
 */

// The corrected flipping function with swapped labels
void flip_for_player2(Quoridor* env, float* obs_buf, int player_id) {
    // Flip walls and player positions
    for (int x = 0; x < WALLS; x++) {
        for (int y = 0; y < WALLS; y++) {
            int fx = WALLS - 1 - x;
            int fy = WALLS - 1 - y;

            // For Player 2's observation:
            // "Me" is player 1 (the current player)
            if (env->players[1].x * 2 == fx && env->players[1].y * 2 == fy)
                *obs_buf++ = -1.0f; // <-- I am -1.0f
            // "Opponent" is player 0
            else if (env->players[0].x * 2 == fx && env->players[0].y * 2 == fy)
                *obs_buf++ = -2.0f; // <-- My opponent is -2.0f
            else
                *obs_buf++ = env->walls[fx][fy];
        }
    }

    // Wall counts must also be from the current player's perspective
    *obs_buf++ = (float)env->players[1].walls; // My walls
    *obs_buf++ = (float)env->players[0].walls; // Opponent's walls
    *obs_buf++ = (float)env->players[player_id].is_turn;
}


// This function is correct as-is for Player 1
void compute_observations(Quoridor* env) {
    int obs_idx = 0;
    for (int a = 0; a < 2; a++) {
        if (a == 0) { // For Player 1
            for (int x = 0; x < WALLS; x++) {
                for (int y = 0; y < WALLS; y++) {
                    // "Me" is player 0
                    if (env->players[0].x * 2 == x && env->players[0].y * 2 == y)
                        env->observations[obs_idx++] = -1.0f;
                    // "Opponent" is player 1
                    else if (env->players[1].x * 2 == x && env->players[1].y * 2 == y)
                        env->observations[obs_idx++] = -2.0f;
                    else
                        env->observations[obs_idx++] = env->walls[x][y];
                }
            }
            env->observations[obs_idx++] = (float)env->players[0].walls;
            env->observations[obs_idx++] = (float)env->players[1].walls;
            env->observations[obs_idx++] = (float)env->players[a].is_turn;
        } else { // For Player 2
            flip_for_player2(env, &env->observations[obs_idx], a);
            obs_idx += WALLS * WALLS + 3; // Ensure this is +3
        }
    }
}



void add_log(Quoridor* env) {
    // Score: 1 for win, 0 for loss, -1 for timeout
    float score = 0.0f;
    if (env->winner == 1) score = 1.0f;
    else if (env->winner == 2) score = 1.0f;
    else score = -1.0f;
    

    env->log.score += (score*env->moves);
    env->log.perf = (((score*env->moves)+200.f) / (400.f));
    env->log.episode_return += env->rewards[0] + env->rewards[1];
    env->log.episode_length += env->moves;
    env->log.n += 1;
}


// Required function
void c_reset(Quoridor* env) {
    env->moves = 0;
    env->winner = 0;
    for (int i = 0; i < WALLS; i++) {
        for (int j = 0; j < WALLS; j++) {
            env->walls[i][j] = EMPTY;
        }
    }

    env->players[0].is_turn = rand() % 2 == 0; // Randomly choose who starts
    env->players[0].walls = INIT_WALLS;
    env->players[0].x = SIZE/2;
    env->players[0].y = 0;

    env->players[1].is_turn = !env->players[0].is_turn;
    env->players[1].walls = INIT_WALLS;
    env->players[1].x = SIZE/2;
    env->players[1].y = SIZE-1;

    env->terminals[0] = 0;
    env->terminals[1] = 0;
    compute_observations(env);
}

// Helper to check if a wall blocks movement between (x1,y1) and (x2,y2)
bool is_wall_blocking(Quoridor* env, int x1, int y1, int x2, int y2) {
    int wx1 = x1 * 2;
    int wy1 = y1 * 2;
    int wx2 = x2 * 2;
    int wy2 = y2 * 2;

    if (x1 == x2 && abs(y1 - y2) == 1) {
        int wall_x = wx1;
        int wall_y = (y1 < y2 ? wy1 : wy2) + 1;
        return wall_x >= 0 && wall_x < WALLS && wall_y >= 0 && wall_y < WALLS &&
               env->walls[wall_x][wall_y] == WALL;
    }
    if (y1 == y2 && abs(x1 - x2) == 1) {
        int wall_x = (x1 < x2 ? wx1 : wx2) + 1;
        int wall_y = wy1;
        return wall_x >= 0 && wall_x < WALLS && wall_y >= 0 && wall_y < WALLS &&
               env->walls[wall_x][wall_y] == WALL;
    }
    return false;
}


// BFS from (sx, sy) to goal_row
bool bfs(Quoridor* env, int sx, int sy, int goal_row, int visited[SIZE][SIZE], int queue[SIZE * SIZE][2]) {
    int front = 0, back = 0;
    memset(visited, 0, sizeof(int) * SIZE * SIZE);
    queue[back][0] = sx;
    queue[back][1] = sy;
    back++;
    visited[sx][sy] = 1;

    while (front < back) {
        int x = queue[front][0];
        int y = queue[front][1];
        front++;

        if (y == goal_row) return true;

        // Try all directions
        for (int d = 0; d < 4; d++) {
            int dx = 0, dy = 0;
            if (d == UP) dy = -1;
            else if (d == DOWN) dy = 1;
            else if (d == LEFT) dx = -1;
            else if (d == RIGHT) dx = 1;

            int nx = x + dx;
            int ny = y + dy;

            // Check bounds
            if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE) continue;

            // ❗ Check wall between (x,y) and (nx,ny)
            if (is_wall_blocking(env, x, y, nx, ny)) continue;

            if (!visited[nx][ny]) {
                visited[nx][ny] = 1;
                queue[back][0] = nx;
                queue[back][1] = ny;
                back++;
            }
        }
    }

    return false;
}


bool is_solvable(Quoridor* env) {
    int visited[SIZE][SIZE];
    int queue[SIZE * SIZE][2];

    // Player 1: must reach y == SIZE-1
    // Player 2: must reach y == 0
    bool p1 = bfs(env, env->players[0].x, env->players[0].y, SIZE - 1, visited, queue);
    bool p2 = bfs(env, env->players[1].x, env->players[1].y, 0, visited, queue);

    return p1 && p2;
}


// Print the wall grid for debugging
void print_wall_grid(Quoridor* env) {
    printf("Wall grid:\n");
    for (int i = 0; i < WALLS; i++) {
        for (int j = 0; j < WALLS; j++) {
            if (env->walls[i][j] == WALL)
                printf("#");
            bool found_player = false;
            for (int x = 0; x < 2; x++) {
                if (env->players[x].x*2 == i && env->players[x].y*2 == j) {
                    printf("%d", x + 1); // Print player number
                    found_player = true;
                    continue;
                }
            }
            if (!found_player && env->walls[i][j] == EMPTY)
                printf(".");
        }
        printf("\n");
    }
    printf("\n");
}


void c_step(Quoridor* env) {
    bool valid_move = true;
    int current = env->players[0].is_turn ? 0 : 1;
    int other = 1 - current;
    int action = env->actions[current];
    env->moves++;
    env->rewards[current] = 0.0f;
    env->rewards[other] = 0.0f;
    if (env->terminals[0] == 1) {
        c_reset(env);
        return;
    }

    // Flip movement and wall placement for player 2 to match its flipped observation
    if (current == 1) {
        if (action >= UP && action <= RIGHT) {
            // Mirror movement direction
            if (action == UP) action = DOWN;
            else if (action == DOWN) action = UP;
            else if (action == LEFT) action = RIGHT;
            else if (action == RIGHT) action = LEFT;
        } else if (action > RIGHT) {
            // This action is a wall placement
            int wall_idx = action - (RIGHT + 1);
            int horizontal_wall_actions = (SIZE - 1) * (SIZE - 1); // 16 for SIZE=5

            if (wall_idx < horizontal_wall_actions) {
                // Horizontal wall
                int wall_x = wall_idx / (SIZE - 1); // 0 to 3
                int wall_y = wall_idx % (SIZE - 1); // 0 to 3
                int flipped_x = (SIZE - 2) - wall_x; // 3 - wall_x (e.g., 0->3, 1->2)
                int flipped_y = (SIZE - 2) - wall_y;
                int flipped_wall_idx = flipped_x * (SIZE - 1) + flipped_y;
                action = flipped_wall_idx + (RIGHT + 1);
            } else {
                // Vertical wall
                int v_idx = wall_idx - horizontal_wall_actions;
                int wall_x = v_idx / (SIZE - 1);
                int wall_y = v_idx % (SIZE - 1);
                int flipped_x = (SIZE - 2) - wall_x;
                int flipped_y = (SIZE - 2) - wall_y;
                int flipped_v_idx = flipped_x * (SIZE - 1) + flipped_y;
                action = flipped_v_idx + horizontal_wall_actions + (RIGHT + 1);
            }
        }
    }


    // Movement and jumping
    if (action >= UP && action <= RIGHT) {
        int dx = 0, dy = 0;
        if (action == UP) dy = -1;
        else if (action == DOWN) dy = 1;
        else if (action == LEFT) dx = -1;
        else if (action == RIGHT) dx = 1;

        int px = env->players[current].x;
        int py = env->players[current].y;
        int nx = px + dx;
        int ny = py + dy;

        if (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE &&
            !is_wall_blocking(env, px, py, nx, ny)) {

            if (nx == env->players[other].x && ny == env->players[other].y) {
                // Jump attempt
                int jx = nx + dx;
                int jy = ny + dy;

                if (jx >= 0 && jx < SIZE && jy >= 0 && jy < SIZE &&
                    !is_wall_blocking(env, nx, ny, jx, jy)) {
                    if (!(jx == px && jy == py)) {
                        env->players[current].x = jx;
                        env->players[current].y = jy;
                    } else {
                        valid_move = false;
                    }
                } else {
                    // Lateral jump attempt
                    int candidates[4][2];
                    int n_candidates = 0;
                    for (int d = 0; d < 4; d++) {
                        int ddx = 0, ddy = 0;
                        if (d == UP) ddy = -1;
                        else if (d == DOWN) ddy = 1;
                        else if (d == LEFT) ddx = -1;
                        else if (d == RIGHT) ddx = 1;
                        int tx = nx + ddx;
                        int ty = ny + ddy;
                        if (tx == px && ty == py) continue;
                        if (tx >= 0 && tx < SIZE && ty >= 0 && ty < SIZE &&
                            !is_wall_blocking(env, nx, ny, tx, ty)) {
                            candidates[n_candidates][0] = tx;
                            candidates[n_candidates][1] = ty;
                            n_candidates++;
                        }
                    }
                    if (n_candidates > 0) {
                        int idx = rand() % n_candidates;
                        env->players[current].x = candidates[idx][0];
                        env->players[current].y = candidates[idx][1];
                    } else {
                        valid_move = false;
                    }
                }
            } else {
                env->players[current].x = nx;
                env->players[current].y = ny;
            }
        } else {
            valid_move = false;
        }

    } else if (action > RIGHT) {

        int wall_idx = action - (RIGHT + 1);
        int max_wall_actions = 2 * (SIZE - 1) * SIZE;

        if (wall_idx >= max_wall_actions) {
            valid_move = false;
        } else if (wall_idx < (SIZE - 1) * SIZE) {
            // Horizontal wall
            int wall_x = wall_idx / (SIZE - 1);
            int wall_y = wall_idx % (SIZE - 1);

            for (int f = 0; f < 3; f++) {
                int wx = 2 * wall_x + f;
                int wy = 2 * wall_y + 1;

                if (wx >= WALLS || wy >= WALLS) {
                    valid_move = false;
                    break;
                }
                if (env->walls[wx][wy] != EMPTY) {
                    valid_move = false;
                    break;
                }
            }

            if (valid_move && env->players[current].walls > 0) {
                for (int f = 0; f < 3; f++) {
                    env->walls[2 * wall_x + f][2 * wall_y + 1] = WALL;
                }
                if (!is_solvable(env)) {
                    for (int f = 0; f < 3; f++) {
                        env->walls[2 * wall_x + f][2 * wall_y + 1] = EMPTY;
                    }
                    valid_move = false;
                } else {
                    env->players[current].walls--;
                }
            } else {
                valid_move = false;
            }

        } else {
            // Vertical wall
            int v_idx = wall_idx - (SIZE - 1) * SIZE;
            int wall_x = v_idx / SIZE;
            int wall_y = v_idx % SIZE;

            for (int f = 0; f < 3; f++) {
                int wx = 2 * wall_x + 1;
                int wy = 2 * wall_y + f;
                if (wx >= WALLS || wy >= WALLS) {
                    valid_move = false;
                    break;
                }
                if (env->walls[wx][wy] != EMPTY) {
                    valid_move = false;
                    break;
                }
            }

            if (valid_move && env->players[current].walls > 0) {
                for (int f = 0; f < 3; f++) {
                    env->walls[2 * wall_x + 1][2 * wall_y + f] = WALL;
                }
                if (!is_solvable(env)) {
                    for (int f = 0; f < 3; f++) {
                        env->walls[2 * wall_x + 1][2 * wall_y + f] = EMPTY;
                    }
                    valid_move = false;
                } else {
                    env->players[current].walls--;
                }
            } else {
                valid_move = false;
            }
        }
    }



    if (env->moves >= 200) {
        // compute who is closer or give it to player whose turn its not

        int player0 = SIZE - 1 - env->players[0].y;
        int player1 = env->players[0].y;

        if (player0 < player1) {
            env->rewards[1] = -1.f;
            env->rewards[0] = 0.f;
        } else if (player1 > player0) {
            env->rewards[0] = 0.f;
            env->rewards[1] = -1.f;
        } else {
            env->rewards[current] = -.5f;
            env->rewards[other] = -.5f;
        }
        env->terminals[0] = 1;
        env->terminals[1] = 1;
        add_log(env);
        return;
    } else if (env->players[current].y == (current == 0 ? SIZE - 1 : 0)) {
        env->rewards[current] = 1.f;
        env->rewards[other] = -1.f;
        env->winner = current + 1;
        env->terminals[0] = 1;
        env->terminals[1] = 1;
        add_log(env);
        return;
    } else if (!valid_move) {
        env->rewards[current] = -0.05f;
    } else {
        env->players[current].is_turn = false;
        env->players[other].is_turn = true;
    }

    compute_observations(env);
}
// Required function. Should handle creating the client on first call
void c_render(Quoridor* env) {
    static bool window_initialized = false;
    const int px = 40;
    const int board_px = WALLS * px;
    static char info_text[64];

    if (!window_initialized) {
        InitWindow(px * WALLS + 200, px * WALLS, "Quoridor");
        SetTargetFPS(15);
        window_initialized = true;
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int i = 0; i < WALLS; i++) {
        for (int j = 0; j < WALLS; j++) {
            if (i % 2 == 0 && j % 2 == 0) {
                DrawRectangle(j * px, i * px, px, px, (Color){18, 72, 180, 255});
            } else if (j >= 0 && j < WALLS && i >= 0 && i < WALLS && env->walls[j][i] == WALL) {
                DrawRectangle(j * px, i * px, px, px, (Color){187, 0, 0, 255});
            } else if (i % 2 == 1 && j % 2 == 1) {
                DrawRectangle(j * px, i * px, px, px, (Color){80, 80, 80, 255});
            }
        }
    }

    for (int p = 0; p < 2; p++) {
        int x = env->players[p].x;
        int y = env->players[p].y;
        //if (x < 0 || x >= WALLS / 2 || y < 0 || y >= WALLS / 2) {
        //    continue;
        //}
        Color pawn_color = (p == 0) ? (Color){187, 0, 0, 255} : (Color){0, 187, 187, 255};
        DrawCircle(x * 2 * px + px / 2, y * 2 * px + px / 2, px / 2 - 6, pawn_color);
    }

    snprintf(info_text, sizeof(info_text), "Player 1 Walls: %d", env->players[0].walls);
    DrawText(info_text, board_px + 10, 40, 20, (Color){241, 241, 241, 241});

    snprintf(info_text, sizeof(info_text), "Player 2 Walls: %d", env->players[1].walls);
    DrawText(info_text, board_px + 10, 70, 20, (Color){241, 241, 241, 241});

    EndDrawing();
}


// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Quoridor* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    // Do NOT call free_allocated(env) here.
}
