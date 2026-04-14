/**
 * Generate reference game traces from Ludii for validation.
 *
 * Usage: javac -cp Ludii.jar generate_trace.java && java -cp .:Ludii.jar generate_trace game.lud [seed] [max_moves]
 *
 * Outputs JSON trace to stdout:
 * {
 *   "game": "Tic-Tac-Toe",
 *   "num_players": 2,
 *   "board_size": 9,
 *   "seed": 42,
 *   "moves": [
 *     {
 *       "move_index": 0,
 *       "mover": 1,
 *       "action_from": -1,
 *       "action_to": 4,
 *       "legal_moves_count": 9,
 *       "legal_to_sites": [0,1,2,3,4,5,6,7,8],
 *       "board_state": [-1,-1,-1,-1,1,-1,-1,-1,-1],
 *       "terminated": false
 *     },
 *     ...
 *   ],
 *   "winner": 1,
 *   "num_moves": 5
 * }
 */

import java.util.*;
import game.Game;
import other.GameLoader;
import other.context.Context;
import other.model.Model;
import other.move.Move;
import other.trial.Trial;
import other.state.container.ContainerState;
import main.collections.FastArrayList;
import game.types.board.SiteType;

public class generate_trace {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: java generate_trace <game.lud> [seed] [max_moves]");
            System.exit(1);
        }

        String gamePath = args[0];
        long seed = args.length > 1 ? Long.parseLong(args[1]) : 42;
        int maxMoves = args.length > 2 ? Integer.parseInt(args[2]) : 200;

        // Load game
        Game game = GameLoader.loadGameFromFile(new java.io.File(gamePath));
        if (game == null) {
            System.err.println("Failed to load game: " + gamePath);
            System.exit(1);
        }

        // Create context and start
        Trial trial = new Trial(game);
        Context context = new Context(game, trial);
        game.start(context);

        Random rng = new Random(seed);

        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"game\": \"").append(game.name()).append("\",\n");
        json.append("  \"num_players\": ").append(game.players().count()).append(",\n");
        json.append("  \"board_size\": ").append(game.board().numSites()).append(",\n");
        json.append("  \"seed\": ").append(seed).append(",\n");
        json.append("  \"moves\": [\n");

        int moveCount = 0;
        while (!trial.over() && moveCount < maxMoves) {
            int mover = context.state().mover();

            // Get legal moves
            FastArrayList<Move> legalMoves = game.moves(context).moves();

            if (legalMoves.isEmpty()) break;

            // Pick random move
            Move chosen = legalMoves.get(rng.nextInt(legalMoves.size()));

            // Record legal move destinations
            Set<Integer> legalTos = new TreeSet<>();
            for (Move m : legalMoves) {
                legalTos.add(m.to());
            }

            // Apply move
            game.apply(context, chosen);

            // Read board state after move
            ContainerState cs = context.containerState(0);
            int numSites = game.board().numSites();
            int[] boardState = new int[numSites];
            for (int i = 0; i < numSites; i++) {
                int who = cs.who(i, SiteType.Cell); // Cell site type
                boardState[i] = who == 0 ? -1 : who; // 0 = empty in Ludii
            }

            // Write move JSON
            if (moveCount > 0) json.append(",\n");
            json.append("    {\n");
            json.append("      \"move_index\": ").append(moveCount).append(",\n");
            json.append("      \"mover\": ").append(mover).append(",\n");
            json.append("      \"action_from\": ").append(chosen.from()).append(",\n");
            json.append("      \"action_to\": ").append(chosen.to()).append(",\n");
            json.append("      \"legal_moves_count\": ").append(legalMoves.size()).append(",\n");
            json.append("      \"legal_to_sites\": ").append(legalTos).append(",\n");
            json.append("      \"board_state\": [");
            for (int i = 0; i < numSites; i++) {
                if (i > 0) json.append(",");
                json.append(boardState[i]);
            }
            json.append("],\n");
            json.append("      \"terminated\": ").append(trial.over()).append("\n");
            json.append("    }");
            moveCount++;
        }

        json.append("\n  ],\n");

        // Final result
        int winner = trial.over() ? context.winners().isEmpty() ? 0 : context.winners().get(0) : 0;
        json.append("  \"winner\": ").append(winner).append(",\n");
        json.append("  \"num_moves\": ").append(moveCount).append("\n");
        json.append("}\n");

        System.out.println(json.toString());
    }
}
