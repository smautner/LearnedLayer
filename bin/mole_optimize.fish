

#  we need to find the path to the py files...
set filepath (dirname (status --current-filename))
set filepath $filepath"/moleLearnedLayer/optimizer.py"
python $filepath $argv


