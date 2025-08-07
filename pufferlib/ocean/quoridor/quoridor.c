#include <time.h>
#include "quoridor.h"
#include "raylib.h"
// Assuming puffernet.h provides the definitions for Weights, Default,
// load_weights, make_default, and forward_default.
#include "puffernet.h"


// The main demonstration function to run the Quoridor game with AI agents.
void demo() {
    Quoridor env;

    // --- FIX: Correct memory allocation for observations ---
    // The original allocation was missing the space for the action masks.
    // Each player's observation consists of the board, metadata, AND the action mask.
    const int obs_board_size = WALLS_GRID_DIM * WALLS_GRID_DIM;
    const int obs_meta_size = 3; // wall counts (2) + turn (1)
    const int obs_per_player = obs_board_size + obs_meta_size + ACTION_SPACE;
    env.observations = calloc(2 * obs_per_player, sizeof(float));

    env.actions = calloc(2, sizeof(int));
    env.rewards = calloc(2, sizeof(float));
    env.terminals = calloc(2, sizeof(unsigned char));
    c_reset(&env);

    // --- FIX: Use the ACTION_SPACE macro from the header file ---
    // The original code used a hardcoded value of 44, which was incorrect.
    // ACTION_SPACE from the .h file is the correct size (36 for SIZE=5).
    int total_actions = ACTION_SPACE;
    printf("Action Space: %d", ACTION_SPACE);
    printf("Observation Space: %d", obs_per_player);

    // These weights would need to be trained with the correct network dimensions.
    // The dimensions passed to make_default must match the training configuration.
    Weights* weights = load_weights("puffer_quoridor_weights.bin", 2458791);
    // --- FIX: Correct input and output dimensions for the neural network ---
    int logit_sizes[1] = {total_actions};
    LinearLSTM* net = make_linearlstm(weights, 2, obs_per_player, logit_sizes, 1, 512);

    //weights = load_weights("puffer_quoridor_weights_old.bin", 152357);
    //LinearLSTM* net2 = make_linearlstm(weights, 2, obs_per_player, logit_sizes, 1, 128);

    const int px = 40;
    // --- FIX: Use the correct macro for board dimensions ---
    // 'WALLS' was not defined; 'WALLS_GRID_DIM' is the correct macro.
    const int board_px = WALLS_GRID_DIM * px;
    InitWindow(board_px + 200, board_px, "Quoridor RL");
    SetTargetFPS(60);
    
    int player_id = 0; // This variable seems unused, the turn is checked via env.players[...].is_turn

    int i = 0;
    int wins_0 = 0;
    int wins_1 = 0;
    int draw = 0;
    while (!WindowShouldClose() && i < 10000) {
        int p1_action = -1;
        bool player_moved = false;

        // Determine which player's turn it is and use the corresponding network.
        //forward_linearlstm(net2, env.observations, env.actions);
        //int a1 = env.actions[0];
        forward_linearlstm(net, env.observations, env.actions);
        //env.actions[0] = a1;

        // This block for manual player input is disabled with '&& false'.
        // The logic inside has been corrected in case it's re-enabled.
        if (env.players[player_id].is_turn && false) {
            while (!player_moved && !WindowShouldClose()) {
                c_render(&env);

                // Movement keys
                if (IsKeyPressed(KEY_UP)) { p1_action = UP; player_moved = true; }
                else if (IsKeyPressed(KEY_DOWN)) { p1_action = DOWN; player_moved = true; }
                else if (IsKeyPressed(KEY_LEFT)) { p1_action = LEFT; player_moved = true; }
                else if (IsKeyPressed(KEY_RIGHT)) { p1_action = RIGHT; player_moved = true; }

                // Mouse input for wall placement
                if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                    Vector2 mouse = GetMousePosition();
                    int mx = (int)(mouse.x) / px;
                    int my = (int)(mouse.y) / px;

                    // --- FIX: Correct logic for wall placement action calculation ---
                    // Horizontal wall: odd row (my), even col (mx)
                    if (my % 2 == 1 && mx % 2 == 0 && mx < WALLS_GRID_DIM - 1) {
                        int c = mx / 2;       // column index for wall
                        int r = (my - 1) / 2; // row index for wall
                        if (c >= 0 && c < SIZE - 1 && r >= 0 && r < SIZE - 1) {
                            p1_action = MOVE_ACTIONS + (r * (SIZE - 1) + c);
                            player_moved = true;
                        }
                    }
                    // Vertical wall: even row (my), odd col (mx)
                    else if (my % 2 == 0 && mx % 2 == 1 && my < WALLS_GRID_DIM - 1) {
                        int c = (mx - 1) / 2; // column index for wall
                        int r = my / 2;       // row index for wall
                        if (c >= 0 && c < SIZE - 1 && r >= 0 && r < SIZE - 1) {
                            p1_action = MOVE_ACTIONS + HORIZONTAL_WALL_PLACEMENTS + (r * (SIZE - 1) + c);
                            player_moved = true;
                        }
                    }
                }

                if (IsKeyPressed(KEY_ESCAPE)) {
                    CloseWindow();
                    exit(0);
                }
            }

            if (p1_action != -1) {
                env.actions[player_id] = p1_action;
            }
        }
        
        c_step(&env);
        c_render(&env);
        
        if (env.terminals[0] || env.terminals[1]) {
            if (env.winner == 1) {
                wins_0++;
            } else if (env.winner == 2) {
                wins_1++;
            } else {
                draw++;
            }
            printf("Game %d: Player 1 wins: %d, Player 2 wins: %d, Draws: %d\n", i+1, wins_0, wins_1, draw);
            if (wins_0 + wins_1 > 0) {
              printf("Win rate P1: %.2f%%\n", (float)wins_0 / (wins_0 + wins_1) * 100.0f);
            }
            // The c_step function automatically calls c_reset if the game is terminal,
            // so the loop can continue with a new game.
        }
        i++;
    }

    c_close(&env); // Clean up the Raylib window
    // Free allocated memory
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    // Assuming puffernet has functions to free these structs
    // destroy_default(net);
    // destroy_default(net2);
    // destroy_weights(weights);
    // destroy_weights(weights2);
}

int main() {
    srand(time(NULL));
    demo();
    return 0;
}