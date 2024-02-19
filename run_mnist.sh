#!/bin/bash

# Name of the tmux session
SESSION_NAME="Mnist_Training"

# Start a new tmux session and create the first pane for digit 0
# Detach (-d) immediately and execute the first command
tmux new-session -d -s $SESSION_NAME

tmux new-window -t $SESSION_NAME -n "Training" 

tmux send-keys -t $SESSION_NAME:Training "echo 'Training digit 0'  && python main.py --digit 0 && python MNIST_test_2.py --digit 0" C-m

# Since we want to keep all panes in a single window, we'll split new panes in this window
for i in {1..9}
do
    # Split the window for each subsequent digit
    # No need to choose between -h and -v; just keep splitting
    tmux split-window -t $SESSION_NAME:Training
    tmux send-keys -t $SESSION_NAME:Training "echo 'Training digit $i' && python main.py --digit $i && python MNIST_test_2.py --digit $i " C-m
    # Rebalance windows after every split to maintain a tiled layout
    tmux select-layout -t $SESSION_NAME:Training tiled
done

# After all splits, make sure the layout is tiled
tmux select-layout -t $SESSION_NAME:Training tiled

# Optionally, synchronize inputs to all panes if needed. Comment out if not needed.
# tmux setw -t $SESSION_NAME:Training synchronize-panes on

# Attach to the session
tmux attach-session -t $SESSION_NAME
