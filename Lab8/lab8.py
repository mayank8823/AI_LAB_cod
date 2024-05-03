from collections import Counter
import random


class State:
    def __init__(self):
        self.state = [" ", " ", " ", " ", " ", " ", " ", " ", " "]

    def __str__(self):
        return (
            "\n 0 | 1 | 2     %s | %s | %s\n"
            "---+---+---   ---+---+---\n"
            " 3 | 4 | 5     %s | %s | %s\n"
            "---+---+---   ---+---+---\n"
            " 6 | 7 | 8     %s | %s | %s"
            % (
                self.state[0],
                self.state[1],
                self.state[2],
                self.state[3],
                self.state[4],
                self.state[5],
                self.state[6],
                self.state[7],
                self.state[8],
            )
        )

    def moves_validity(self, move):
        try:
            move = int(move)
        except ValueError:
            return False
        if 0 <= move <= 8 and self.state[move] == " ":
            return True
        return False

    def won_state(self):
        return (
            (
                self.state[0] != " "
                and (
                    (self.state[0] == self.state[1] == self.state[2])
                    or (self.state[0] == self.state[3] == self.state[6])
                    or (self.state[0] == self.state[4] == self.state[8])
                )
            )
            or (
                self.state[4] != " "
                and (
                    (self.state[1] == self.state[4] == self.state[7])
                    or (self.state[3] == self.state[4] == self.state[5])
                    or (self.state[2] == self.state[4] == self.state[6])
                )
            )
            or (
                self.state[8] != " "
                and (
                    (self.state[2] == self.state[5] == self.state[8])
                    or (self.state[6] == self.state[7] == self.state[8])
                )
            )
        )

    def draw_state(self):
        return all((x != " " for x in self.state))

    def move(self, ps, sign):
        self.state[ps] = sign

    def s(self):
        return "".join(self.state)


class MenacePlayer:
    def __init__(self):
        self.box = {}
        self.win_count = 0
        self.draw_count = 0
        self.lose_count = 0

    def start(self):
        self.action = []

    def nextAction(self, state):
        state = state.s()
        if state not in self.box:
            nextbead = [pos for pos, mark in enumerate(state) if mark == " "]
            self.box[state] = nextbead * ((len(nextbead) + 2) // 2)
        beads = self.box[state]
        if len(beads):
            bead = random.choice(beads)
            self.action.append((state, bead))
        else:
            bead = -1
        return bead

    def winGame(self):
        for state, bead in self.action:
            self.box[state].extend([bead, bead, bead])
        self.win_count += 1

    def drawGame(self):
        for state, bead in self.action:
            self.box[state].append(bead)
        self.draw_count += 1

    def loseGame(self):
        for state, bead in self.action:
            boxes = self.box[state]
            del boxes[boxes.index(bead)]
        self.lose_count += 1

    def print_stats(self):
        print("Total game learnt %d boards" % len(self.box))
        print("Won : %d" % self.win_count)
        print("Draw : %d" % self.draw_count)
        print("Lose : %d" % self.lose_count)

    def length(self):
        return len(self.box)


class HumanPlayer:
    def __init__(self):
        pass

    def start(self):
        print("Get ready!")

    def nextAction(self, board):
        while True:
            turn = input("Your turn: ")
            if board.moves_validity(turn):
                break
            print("Not a valid turn")
        return int(turn)

    def winGame(self):
        print("You won!")

    def drawGame(self):
        print("It's a draw.")

    def loseGame(self):
        print("You lose.")

    def print_probability(self, board):
        pass


def playGame(first, second, silent=False):
    first.start()
    second.start()
    board = State()
    if not silent:
        print("\n\nStarting a new game!")
        print(board)
    while True:
        move = first.nextAction(board)
        if move == -1:
            if not silent:
                print("Player resigns")
                print(board)
            first.loseGame()
            second.winGame()
            break
        board.move(move, "X")
        if not silent:
            print(board)
        if board.won_state():
            first.winGame()
            second.loseGame()
            break
        if board.draw_state():
            first.drawGame()
            second.drawGame()
            break
        move = second.nextAction(board)
        if move == -1:
            if not silent:
                print("Player resigns")
                print(board)
            second.loseGame()
            first.winGame()
            break
        board.move(move, "O")
        if not silent:
            print(board)
        if board.won_state():
            second.winGame()
            first.loseGame()
            break


if __name__ == "__main__":
    go_first_menace = MenacePlayer()
    go_second_menace = MenacePlayer()
    human = HumanPlayer()
    for i in range(1000):
        playGame(go_first_menace, go_second_menace, silent=True)
    go_first_menace.print_stats()
    go_second_menace.print_stats()
    playGame(human, go_second_menace)
