import math


def minmax(intial_configuration, is_maxAgent):
    if all(x == 0 for x in intial_configuration):
        return (0, []) if is_maxAgent else (1, [])

    best_score = -math.inf if is_maxAgent else math.inf
    best_moves = []

    for i, pile in enumerate(intial_configuration):
        for j in range(1, pile + 1):
            new_configuration = intial_configuration[:]
            new_configuration[i] = pile - j
            current_score, moves = minmax(new_configuration, not is_maxAgent)
            if is_maxAgent:
                if current_score > best_score:
                    best_score = current_score
                    best_moves = [(i, j)] + moves
            else:
                if current_score < best_score:
                    best_score = current_score
                    best_moves = [(i, j)] + moves

    return (best_score, best_moves)


def main():
    intial_configuration = list(
        map(
            int,
            input(
                "Enter the number of objects in each Piles, separated by spaces: "
            ).split(),
        )
    )
    turn = True  # false => Player 1

    while True:
        if turn:
            print("Current state of the game:", intial_configuration)
            pile_select = (
                int(
                    input(
                        " Player Select Pile? (between 1 and %d): "
                        % len(intial_configuration)
                    )
                )
                - 1
            )
            if pile_select < 0 or pile_select >= len(intial_configuration):
                print("Invalid Pile. Please try again.")
                continue
            remove = int(
                input(
                    "Player No of objects to remove? (between 1 and %d): "
                    % intial_configuration[pile_select]
                )
            )
            if remove < 1 or remove > intial_configuration[pile_select]:
                print("Invalid input. Please try again.")
                continue
            intial_configuration[pile_select] -= remove
            turn = False
            if all(x == 0 for x in intial_configuration):
                print("You win!")
                return
        else:
            print("Current state of the game:", intial_configuration)
            current_score, moves = minmax(intial_configuration, True)
            pile_select, remove = moves[0]
            intial_configuration[pile_select] -= remove
            print("Computer removed", remove, "objects from pile", pile_select + 1)
            print("Current state of the game:", intial_configuration)
            turn = True
            if all(x == 0 for x in intial_configuration):
                print("Computer win!")
                return


main()


# %%
import math


def minmax(intial_configuration, is_maxAgent):
    if all(x == 0 for x in intial_configuration):
        return (0, []) if is_maxAgent else (1, [])

    best_score = -math.inf if is_maxAgent else math.inf
    best_moves = []

    for i, pile in enumerate(intial_configuration):
        for j in range(1, pile + 1):
            new_configuration = intial_configuration[:]
            new_configuration[i] = pile - j
            current_score, moves = minmax(new_configuration, not is_maxAgent)
            if is_maxAgent:
                if current_score > best_score:
                    best_score = current_score
                    best_moves = [(i, j)] + moves
            else:
                if current_score < best_score:
                    best_score = current_score
                    best_moves = [(i, j)] + moves

    return (best_score, best_moves)


def main():
    intial_configuration = list(
        map(
            int,
            input(
                "Enter the number of objects in each Piles, separated by spaces: "
            ).split(),
        )
    )
    turn = True  # false => Player 1

    while True:
        if turn:
            print("Current state of the game:", intial_configuration)
            current_score, moves = minmax(intial_configuration, False)
            pile_select, remove = moves[0]
            intial_configuration[pile_select] -= remove
            print("Computer removed", remove, "objects from pile", pile_select + 1)
            print("Current state of the game:", intial_configuration)
            turn = False
            if all(x == 0 for x in intial_configuration):
                print("Computer 1 win!")
                return
        else:
            print("Current state of the game:", intial_configuration)
            current_score, moves = minmax(intial_configuration, True)
            pile_select, remove = moves[0]
            intial_configuration[pile_select] -= remove
            print("Computer removed", remove, "objects from pile", pile_select + 1)
            print("Current state of the game:", intial_configuration)
            turn = True
            if all(x == 0 for x in intial_configuration):
                print("Computer 2 win!")
                return


main()
