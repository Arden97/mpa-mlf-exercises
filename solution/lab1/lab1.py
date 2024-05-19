import numpy as np
import random
import matplotlib.pyplot as plt
import time
import pandas as pd

x = "X "
revert = False
reverse_i = 2
for i in range(1, 10, 1):
    if not revert:
        print(i*x)
        if i == 5:
            revert = True
    else:
        print((i-reverse_i)*x)
        reverse_i+=2

input_str = "n45as29@#8ss6"
y = []
for i in range(0, len(input_str),1):
    c = input_str[i]
    if c.isnumeric():
        y.append(int(input_str[i]))
print(sum(y))

number = 10
b = []
while number >= 1:
  number = number // 2
  print(number)
  b.append(str(number % 2))
print("".join(b))

def fibonaci(upper_threshold: int) -> list:
    i = 0
    l = [0]
    while l[-1] < upper_threshold:
        if l[-1] == 0:
            l.append(1)
            i+=2
        else:
            l.append(l[i-2]+l[i-1])
            i+=1
    return l[:-1]
print(fibonaci(10))

def rock_paper_scissors(n) -> None:
    my_score = 0
    player_score = 0
    round = 1
    while round <= n:
        player_move = input(f"Round {round}! Make a move: ").lower()
        my_move = random.randint(1, 3)
        if my_move == 1:
            my_move = "rock"
        elif my_move == 2:
            my_move = "paper"
        elif my_move == 3:
            my_move = "scissors"

        print(f"My move is {my_move}")

        if my_move == player_move:
            print("It's a tie")
        elif (my_move == "rock" and player_move == "paper" or 
            my_move == "paper" and player_move == "scissors" or
            my_move == "scissors" and player_move == "rock"):
            print('You win')
            player_score += 1
        else:
            print('You lose')
            my_score += 1
        round+=1

    print(f"End of the game!\nYour score {player_score}\nMy score {my_score}")
    return

#rock_paper_scissors(3)

# arr5x5 = np.array([random.randint(25, size=(5)),
#                 random.randint(25, size=(5)),
#                 random.randint(25, size=(5)),
#                 random.randint(25, size=(5)),
#                 random.randint(25, size=(5))])

def set_to_0(arr, threshold):
    pass


df = pd.read_csv('data.csv')

for x in df.index:
  if df.loc[x, "total_bedrooms"] > 120:
    print(df.loc(x))

#remember to include the 'inplace = True' argument to make the changes in the original DataFrame object instead of returning a copy

print(df.to_string())

