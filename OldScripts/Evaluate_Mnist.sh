# Name of the tmux session
SESSION_NAME="Mnist_evaluating"

# Start a new tmux session and create the first pane for digit 0
# Detach (-d) immediately and execute the first command
tmux new-session -d -s $SESSION_NAME -n "Training" 

# Send initial echo command to the first window. Assuming you want to run something here.
tmux send-keys -t $SESSION_NAME:Training "echo 'Starting eval Script for digit 0' && python Evaluate_Best_Models.py --digit 0 --nbr_samples 10000" C-m

# Since we want to keep all panes in a single window, we'll split new panes in this window
for i in {1..9}
do
    # Split the window for each subsequent digit
    # No need to choose between -h and -v; just keep splitting
    tmux split-window -t $SESSION_NAME:Training
    tmux send-keys -t $SESSION_NAME:Training "echo 'evaluating digit $i' && python Evaluate_Best_Models.py --digit $i --nbr_samples 10000" C-m
    # Rebalance windows after every split to maintain a tiled layout
    tmux select-layout -t $SESSION_NAME:Training tiled
done

# After all splits, make sure the layout is tiled
tmux select-layout -t $SESSION_NAME:Training tiled

# Optionally, synchronize inputs to all panes if needed. Comment out if not needed.
# tmux setw -t $SESSION_NAME:Training synchronize-panes on

# Attach to the session
tmux attach-session -t $SESSION_NAME
