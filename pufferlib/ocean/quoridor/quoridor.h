#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "raylib.h"

#define SIZE 9
#define INIT_WALLS 10

#define EMPTY 0
#define WALL 1
#define WALLS_GRID_DIM (SIZE * 2 - 1)

// Action definitions
#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3
#define MOVE_ACTIONS 4

// Action Space Calculation
#define HORIZONTAL_WALL_PLACEMENTS ((SIZE - 1) * (SIZE - 1))
#define VERTICAL_WALL_PLACEMENTS ((SIZE - 1) * (SIZE - 1))
#define TOTAL_WALL_ACTIONS (HORIZONTAL_WALL_PLACEMENTS + VERTICAL_WALL_PLACEMENTS)
#define ACTION_SPACE (MOVE_ACTIONS + TOTAL_WALL_ACTIONS)

#define MAX_PATH_LENGTH (SIZE * SIZE)

// Required struct for logging.
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    int x;
    int y;
    int walls;
    bool is_turn;
} Player;

// Environment struct for Quoridor
typedef struct {
    Log log;
    Player players[2];
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int moves;
    int winner;
    float walls[WALLS_GRID_DIM][WALLS_GRID_DIM];

    // --- CACHING IMPLEMENTATION ---
    int cached_path_p1[MAX_PATH_LENGTH][2];
    int path_length_p1;
    int cached_path_p2[MAX_PATH_LENGTH][2];
    int path_length_p2;
    
    // --- OPTIMIZATION: BFS VISITED ARRAY ---
    int visited[SIZE][SIZE];
    int visited_generation;

} Quoridor;

// Forward declarations
void compute_action_mask(Quoridor* env, int player_id, float* mask_out);
bool is_solvable(Quoridor* env);
bool bfs_and_cache_path(Quoridor* env, int player_id);
static inline bool is_wall_blocking(Quoridor* env, int x1, int y1, int x2, int y2);
bool does_wall_intersect_cached_paths(Quoridor* env);


// Flips the board state for Player 2's perspective
void flip_for_player2(Quoridor* env, float* obs_buf) {
    for (int x = 0; x < WALLS_GRID_DIM; x++) {
        for (int y = 0; y < WALLS_GRID_DIM; y++) {
            int flipped_y = WALLS_GRID_DIM - 1 - y;
            int flipped_x = WALLS_GRID_DIM - 1 - x;
            
            float* obs_cell = &obs_buf[y * WALLS_GRID_DIM + x];

            if (env->players[1].x * 2 == flipped_x && env->players[1].y * 2 == flipped_y)
                *obs_cell = -1.0f; 
            else if (env->players[0].x * 2 == flipped_x && env->players[0].y * 2 == flipped_y)
                *obs_cell = -2.0f;
            else
                *obs_cell = env->walls[flipped_x][flipped_y];
        }
    }
}


void compute_observations(Quoridor* env) {
    const int obs_board_size = WALLS_GRID_DIM * WALLS_GRID_DIM;
    const int obs_meta_size = 3; // wall counts (2) + turn (1)
    const int obs_core_size = obs_board_size + obs_meta_size;
    const int obs_per_player = obs_core_size + ACTION_SPACE;

    for (int player_id = 0; player_id < 2; player_id++) {
        float* obs_buf_start = &env->observations[player_id * obs_per_player];
        float* obs_buf_board = obs_buf_start;
        
        if (player_id == 0) {
            for (int x = 0; x < WALLS_GRID_DIM; x++) {
                for (int y = 0; y < WALLS_GRID_DIM; y++) {
                    float* obs_cell = &obs_buf_board[y * WALLS_GRID_DIM + x];
                    if (env->players[0].x * 2 == x && env->players[0].y * 2 == y)
                        *obs_cell = -1.0f;
                    else if (env->players[1].x * 2 == x && env->players[1].y * 2 == y)
                        *obs_cell = -2.0f;
                    else
                        *obs_cell = env->walls[x][y];
                }
            }
        } else {
            flip_for_player2(env, obs_buf_board);
        }

        float* obs_buf_meta = obs_buf_start + obs_board_size;
        int me = player_id;
        int opp = 1 - player_id;
        *obs_buf_meta++ = (float)env->players[me].walls;
        *obs_buf_meta++ = (float)env->players[opp].walls;
        *obs_buf_meta++ = (float)env->players[me].is_turn;

        if (env->players[player_id].is_turn) {
            compute_action_mask(env, player_id, obs_buf_meta);
        } else {
            for (int i = 0; i < ACTION_SPACE; i++) {
                *obs_buf_meta++ = 0.0f;
            }
        }
    }
}


void add_log(Quoridor* env) {
    float score = 0.f;
    if (env->winner == 1 || env->winner == 2)
     score = (200.f - env->moves) / 200.f;
    
    env->log.score += score;
    env->log.perf += score;
    env->log.episode_return += env->rewards[0];
    env->log.episode_length += env->moves;
    env->log.n += 1;
}

void c_reset(Quoridor* env) {
    env->moves = 0;
    env->winner = 0;
    memset(env->walls, EMPTY, sizeof(env->walls));

    env->players[0].is_turn = rand() % 2 == 0;
    env->players[0].walls = INIT_WALLS;
    env->players[0].x = SIZE / 2;
    env->players[0].y = 0;

    env->players[1].is_turn = !env->players[0].is_turn;
    env->players[1].walls = INIT_WALLS;
    env->players[1].x = SIZE / 2;
    env->players[1].y = SIZE - 1;

    memset(env->visited, 0, sizeof(env->visited));
    env->visited_generation = 0;

    bfs_and_cache_path(env, 0);
    bfs_and_cache_path(env, 1);

    env->terminals[0] = 0;
    env->terminals[1] = 0;
    compute_observations(env);
}

static inline bool is_wall_blocking(Quoridor* env, int x1, int y1, int x2, int y2) {
    if (x1 == x2 && abs(y1 - y2) == 1) {
        int wall_y = (y1 < y2 ? y1 * 2 : y2 * 2) + 1;
        int wall_x = x1 * 2;
        return env->walls[wall_x][wall_y] == WALL;
    }
    if (y1 == y2 && abs(x1 - x2) == 1) {
        int wall_x = (x1 < x2 ? x1 * 2 : x2 * 2) + 1;
        int wall_y = y1 * 2;
        return env->walls[wall_x][wall_y] == WALL;
    }
    return false;
}

bool bfs(Quoridor* env, int sx, int sy, int goal_row) {
    env->visited_generation++;
    int current_gen = env->visited_generation;
    int queue[SIZE * SIZE][2];
    int front = 0, back = 0;

    queue[back][0] = sx;
    queue[back][1] = sy;
    back++;
    env->visited[sx][sy] = current_gen;

    while (front < back) {
        int x = queue[front][0];
        int y = queue[front][1];
        front++;

        if (y == goal_row) return true;

        int dx[] = {0, 0, -1, 1};
        int dy[] = {-1, 1, 0, 0};

        for (int d = 0; d < 4; d++) {
            int nx = x + dx[d];
            int ny = y + dy[d];
            if (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && env->visited[nx][ny] != current_gen) {
                if (!is_wall_blocking(env, x, y, nx, ny)) {
                    env->visited[nx][ny] = current_gen;
                    queue[back][0] = nx;
                    queue[back][1] = ny;
                    back++;
                }
            }
        }
    }
    return false;
}

bool bfs_and_cache_path(Quoridor* env, int player_id) {
    Player* p = &env->players[player_id];
    int goal_row = (player_id == 0) ? SIZE - 1 : 0;
    int (*path_cache)[2] = (player_id == 0) ? env->cached_path_p1 : env->cached_path_p2;
    int* path_length = (player_id == 0) ? &env->path_length_p1 : &env->path_length_p2;
    *path_length = 0;

    env->visited_generation++;
    int current_gen = env->visited_generation;
    
    int parent[SIZE][SIZE][2] = {0};
    int queue[SIZE * SIZE][2];
    int front = 0, back = 0;

    queue[back][0] = p->x;
    queue[back][1] = p->y;
    back++;
    env->visited[p->x][p->y] = current_gen;
    parent[p->x][p->y][0] = -1; 

    int end_x = -1, end_y = -1;

    while (front < back) {
        int x = queue[front][0];
        int y = queue[front][1];
        front++;

        if (y == goal_row) { end_x = x; end_y = y; break; }

        int dx[] = {0, 0, -1, 1};
        int dy[] = {-1, 1, 0, 0};

        for (int d = 0; d < 4; d++) {
            int nx = x + dx[d], ny = y + dy[d];
            if (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && env->visited[nx][ny] != current_gen) {
                if (!is_wall_blocking(env, x, y, nx, ny)) {
                    env->visited[nx][ny] = current_gen;
                    parent[nx][ny][0] = x;
                    parent[nx][ny][1] = y;
                    queue[back][0] = nx;
                    queue[back][1] = ny;
                    back++;
                }
            }
        }
    }
    
    if (end_x != -1) {
        int cx = end_x, cy = end_y, path_idx = 0;
        while(cx != -1 && path_idx < MAX_PATH_LENGTH) {
            path_cache[path_idx][0] = cx; path_cache[path_idx][1] = cy;
            path_idx++;
            int next_x = parent[cx][cy][0]; int next_y = parent[cx][cy][1];
            cx = next_x; cy = next_y;
        }
        *path_length = path_idx;
        for(int i = 0; i < (*path_length) / 2; i++) {
            int temp_x = path_cache[i][0], temp_y = path_cache[i][1];
            path_cache[i][0] = path_cache[(*path_length) - 1 - i][0];
            path_cache[i][1] = path_cache[(*path_length) - 1 - i][1];
            path_cache[(*path_length) - 1 - i][0] = temp_x;
            path_cache[(*path_length) - 1 - i][1] = temp_y;
        }
        return true;
    }
    return false;
}

bool is_solvable(Quoridor* env) {
    bool p1_can_win = bfs(env, env->players[0].x, env->players[0].y, SIZE - 1);
    bool p2_can_win = bfs(env, env->players[1].x, env->players[1].y, 0);
    return p1_can_win && p2_can_win;
}

void c_step(Quoridor* env) {
    if (env->terminals[0] == 1) { c_reset(env); return; }

    bool valid_move = true;
    int current = env->players[0].is_turn ? 0 : 1;
    int other = 1 - current;
    int action = env->actions[current];

    env->moves++;
    env->rewards[current] = 0.0f;
    env->rewards[other] = 0.0f;

    if (action < MOVE_ACTIONS) {
        int dx = 0, dy = 0;
        if (action == UP) dy = -1; else if (action == DOWN) dy = 1;
        else if (action == LEFT) dx = -1; else if (action == RIGHT) dx = 1;

        int px = env->players[current].x, py = env->players[current].y;
        int nx = px + dx, ny = py + dy;

        if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE || is_wall_blocking(env, px, py, nx, ny)) {
            valid_move = false;
        } else if (nx == env->players[other].x && ny == env->players[other].y) {
            int jx = nx + dx, jy = ny + dy;
            if (jx >= 0 && jx < SIZE && jy >= 0 && jy < SIZE && !is_wall_blocking(env, nx, ny, jx, jy)) {
                env->players[current].x = jx; env->players[current].y = jy;
            } else {
                bool jumped = false;
                if (dx == 0) { // Vertical move
                    if (nx-1 >= 0 && !is_wall_blocking(env, nx, ny, nx-1,ny)) { env->players[current].x = nx-1; env->players[current].y = ny; jumped = true;}
                    else if (nx+1 < SIZE && !is_wall_blocking(env, nx, ny, nx+1,ny)) { env->players[current].x = nx+1; env->players[current].y = ny; jumped = true;}
                } else { // Horizontal move
                    if (ny-1 >= 0 && !is_wall_blocking(env, nx, ny, nx,ny-1)) { env->players[current].x = nx; env->players[current].y = ny-1; jumped = true;}
                    else if (ny+1 < SIZE && !is_wall_blocking(env, nx, ny, nx,ny+1)) { env->players[current].x = nx; env->players[current].y = ny+1; jumped = true;}
                }
                if (!jumped) valid_move = false;
            }
        } else {
            env->players[current].x = nx; env->players[current].y = ny;
        }
    } else { // Wall Placement
        int wall_action_idx = action - MOVE_ACTIONS;
        if (env->players[current].walls <= 0 || wall_action_idx >= TOTAL_WALL_ACTIONS) {
            valid_move = false;
        } else {
            bool is_horizontal = wall_action_idx < HORIZONTAL_WALL_PLACEMENTS;
            int idx = is_horizontal ? wall_action_idx : wall_action_idx - HORIZONTAL_WALL_PLACEMENTS;
            int r = idx / (SIZE - 1), c = idx % (SIZE - 1);
            int w1x, w1y, w2x, w2y, ix, iy;
            if (is_horizontal) { w1x = 2*c; w1y = 2*r+1; w2x = 2*c+2; w2y = 2*r+1; ix = 2*c+1; iy = w1y; } 
            else { w1x = 2*c+1; w1y = 2*r; w2x = 2*c+1; w2y = 2*r+2; ix = w1x; iy = 2*r+1; }
            int w3x = is_horizontal ? w2x : w1x; int w3y = is_horizontal ? w1y : w2y;
            
            if (env->walls[w1x][w1y] || env->walls[w3x][w3y] || env->walls[ix][iy]) {
                valid_move = false;
            } else {
                env->walls[w1x][w1y] = WALL; env->walls[w3x][w3y] = WALL; env->walls[ix][iy] = WALL;
                if (!is_solvable(env)) {
                    env->walls[w1x][w1y] = EMPTY; env->walls[w3x][w3y] = EMPTY; env->walls[ix][iy] = EMPTY;
                    valid_move = false;
                } else {
                    env->players[current].walls--;
                }
            }
        }
    }

    if (!valid_move) {
        env->rewards[current] = -1.0f; env->rewards[other] = 0.0f;
        env->terminals[0] = 1; env->terminals[1] = 1;
        add_log(env);
    } else {
        bfs_and_cache_path(env, 0);
        bfs_and_cache_path(env, 1);
        
        if (env->players[current].y == (current == 0 ? SIZE - 1 : 0)) {
            env->winner = current + 1;
            env->rewards[current] = .5f; env->rewards[other] = -.5f;
            env->terminals[0] = 1; env->terminals[1] = 1;
            add_log(env);
        } else if (env->moves >= 200) {
            env->winner = 0;
            env->rewards[0] = 0.0f; env->rewards[1] = 0.0f;
            env->terminals[0] = 1; env->terminals[1] = 1;
            add_log(env);
        } else {
            env->players[current].is_turn = false;
            env->players[other].is_turn = true;
        }
    }
    compute_observations(env);
}

void c_render(Quoridor* env) {
    static bool window_initialized = false;
    const int px = 40;
    const int board_px = WALLS_GRID_DIM * px;
    if (!window_initialized) { InitWindow(board_px + 200, board_px, "Quoridor"); SetTargetFPS(15); window_initialized = true; }
    if (WindowShouldClose() || IsKeyPressed(KEY_ESCAPE)) { if (IsWindowReady()) CloseWindow(); exit(0); }
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    for (int i = 0; i < WALLS_GRID_DIM; i++) {
        for (int j = 0; j < WALLS_GRID_DIM; j++) {
            if (i % 2 == 0 && j % 2 == 0) DrawRectangle(j * px, i * px, px, px, (Color){18, 72, 180, 255});
            else if (env->walls[j][i] == WALL) DrawRectangle(j * px, i * px, px, px, (Color){187, 0, 0, 255});
            else if (i % 2 == 1 && j % 2 == 1) DrawRectangle(j * px, i * px, px, px, (Color){80, 80, 80, 255});
        }
    }
    for (int p = 0; p < 2; p++) {
        Color pawn_color = (p == 0) ? (Color){220, 220, 170, 255} : (Color){0, 187, 187, 255};
        DrawCircle(env->players[p].x * 2 * px + px/2, env->players[p].y * 2 * px + px/2, px / 2.0f - 4, pawn_color);
    }
    char info_text[64];
    snprintf(info_text, sizeof(info_text), "P1 Walls: %d", env->players[0].walls);
    DrawText(info_text, board_px + 10, 40, 20, RAYWHITE);
    snprintf(info_text, sizeof(info_text), "P2 Walls: %d", env->players[1].walls);
    DrawText(info_text, board_px + 10, 70, 20, RAYWHITE);
    snprintf(info_text, sizeof(info_text), "Turn: %s", env->players[0].is_turn ? "Player 1" : "Player 2");
    DrawText(info_text, board_px + 10, 100, 20, RAYWHITE);
    EndDrawing();
}

void c_close(Quoridor* env) { if (IsWindowReady()) { CloseWindow(); }}

/**
 * @brief Checks if a hypothetical wall (already placed on the board) intersects with any cached path.
 * @param env The game state, where a temporary wall has already been placed.
 * @return true if the wall lies on a cached path, false otherwise.
 */
bool does_wall_intersect_cached_paths(Quoridor* env) {
    // Check player 1's cached path
    for (int i = 0; i < env->path_length_p1 - 1; ++i) {
        if (is_wall_blocking(env, env->cached_path_p1[i][0], env->cached_path_p1[i][1], env->cached_path_p1[i+1][0], env->cached_path_p1[i+1][1])) {
            return true; // The wall blocks this path segment.
        }
    }
    // Check player 2's cached path
    for (int i = 0; i < env->path_length_p2 - 1; ++i) {
        if (is_wall_blocking(env, env->cached_path_p2[i][0], env->cached_path_p2[i][1], env->cached_path_p2[i+1][0], env->cached_path_p2[i+1][1])) {
            return true; // The wall blocks this path segment.
        }
    }
    return false; // The wall does not block any known shortest path.
}

bool is_wall(Quoridor* env, int w1x, int w1y) {
    if (w1x < 0 || w1x >= WALLS_GRID_DIM) return true;
    if (w1y < 0 || w1y >= WALLS_GRID_DIM) return true;
    return env->walls[w1x][w1y] == WALL;
}

bool unblocking_heuristic(Quoridor* env, int w1x, int w1y, int w3x, int w3y, int ix, int iy) {
    int c = 0;

    // horizontal
    if (w1y == w3y) {
        int d0[] = {w1x-2, w1x-1, w1x-1, ix, ix, w3x+2, w3x+1, w3x+1};
        int d1[] = {w1y, w1y+1, w1y-1, iy+1, iy-1, w3y, w3y-1, w3y+1};

        for (int i = 0; i < 8; i++) {
            if (is_wall(env, d0[i], d1[i])) {
                c+=1;
            }
            if (c >= 2) {
                return true;
            }
        }

    } else {
        int d0[] = {w1y-2, w1y-1, w1y-1, iy, iy, w3y+2, w3y+1, w3y+1};
        int d1[] = {w1x, w1x+1, w1x-1, ix+1, ix-1, w3x, w3x-1, w3x+1};

        for (int i = 0; i < 8; i++) {
            if (is_wall(env, d1[i], d0[i])) {
                c+=1;
            }
            if (c >= 2) {
                return true;
            }
        }
        
    }

    return false;

    // vertical
}

void compute_action_mask(Quoridor* env, int player_id, float* mask_out) {
    Player* me = &env->players[player_id];
    Player* opp = &env->players[1 - player_id];
    int mask_idx = 0;

    // 1. Movement Actions
    int dx[] = {0, 0, -1, 1};
    int dy[] = {-1, 1, 0, 0};
    for (int d = 0; d < MOVE_ACTIONS; d++) {
        int nx = me->x + dx[d], ny = me->y + dy[d];
        bool move_is_valid = true;
        if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE || is_wall_blocking(env, me->x, me->y, nx, ny)) {
            move_is_valid = false;
        } else if (nx == opp->x && ny == opp->y) {
            move_is_valid = false;
            int jx = opp->x + dx[d], jy = opp->y + dy[d];
            if (jx >= 0 && jx < SIZE && jy >= 0 && jy < SIZE && !is_wall_blocking(env, opp->x, opp->y, jx, jy)) {
                move_is_valid = true;
            } else {
                 if (dx[d] == 0) { // Vertical move
                    if ((opp->x-1 >= 0 && !is_wall_blocking(env, opp->x, opp->y, opp->x-1, opp->y)) ||
                        (opp->x+1 < SIZE && !is_wall_blocking(env, opp->x, opp->y, opp->x+1, opp->y))) move_is_valid = true;
                } else { // Horizontal move
                    if ((opp->y-1 >= 0 && !is_wall_blocking(env, opp->x, opp->y, opp->x, opp->y-1)) ||
                        (opp->y+1 < SIZE && !is_wall_blocking(env, opp->x, opp->y, opp->x, opp->y+1))) move_is_valid = true;
                }
            }
        }
        mask_out[mask_idx++] = move_is_valid ? 1.0f : 0.0f;
    }

    // 2. Wall Placement Actions
    if (me->walls <= 0) {
        for (int i = 0; i < TOTAL_WALL_ACTIONS; i++) mask_out[mask_idx++] = 0.0f;
        return;
    }

    for (int i = 0; i < TOTAL_WALL_ACTIONS; i++) {
        bool is_horizontal = i < HORIZONTAL_WALL_PLACEMENTS;
        int wall_idx = is_horizontal ? i : i - HORIZONTAL_WALL_PLACEMENTS;
        int r = wall_idx / (SIZE - 1), c = wall_idx % (SIZE - 1);
        int w1x, w1y, w2x, w2y, ix, iy;
        if (is_horizontal) { w1x = 2*c; w1y = 2*r+1; w2x = 2*c+2; ix = 2*c+1; iy = w1y; } 
        else { w1x = 2*c+1; w1y = 2*r; w2y = 2*r+2; ix = w1x; iy = 2*r+1; }
        int w3x = is_horizontal ? w2x : w1x; int w3y = is_horizontal ? w1y : w2y;

        if (env->walls[w1x][w1y] || env->walls[w3x][w3y] || env->walls[ix][iy]) {
            mask_out[mask_idx++] = 0.0f;
            continue;
        }
        
        // --- CACHING LOGIC ---
        env->walls[w1x][w1y] = WALL; env->walls[w3x][w3y] = WALL; env->walls[ix][iy] = WALL;

        if (!does_wall_intersect_cached_paths(env)) {
            mask_out[mask_idx++] = 1.0f;
        } else {
            if (unblocking_heuristic(env, w1x, w1y, w3x, w3y, ix, iy) && !is_solvable(env)) {
                mask_out[mask_idx++] = 0.0f;
            } else {
                mask_out[mask_idx++] = 1.0f;
            }
        }
        
        env->walls[w1x][w1y] = EMPTY; env->walls[w3x][w3y] = EMPTY; env->walls[ix][iy] = EMPTY;
    }
}